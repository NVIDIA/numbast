# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import math
import glob

import cffi
import pytest

import numpy as np
from numpy.testing import assert_allclose

from numba import cuda

from curand_device import (
    curand_init,
    curand,
    curand_uniform,
    curand_uniform_double,
    curand_normal,
    curand_poisson,
    curandStatesXORWOW,
    curandStatesMRG32k3a,
    curandStatesPhilox4_32_10,
    curandStatesScrambledSobol64,
    states_arg_handlers,
    get_shims,
)

from curand_host import (
    curandGetDirectionVectors64,
    curandGetScrambleConstants64,
    curandDirectionVectorSet,
)

# Various parameters

threads = 64
blocks = 64
nthreads = blocks * threads

sample_count = 10000
repetitions = 50


@pytest.mark.parametrize(
    "States",
    [curandStatesXORWOW, curandStatesMRG32k3a, curandStatesPhilox4_32_10],
)
def test_curand_int(States):
    # State initialization kernel
    @cuda.jit(link=get_shims(), extensions=states_arg_handlers)
    def setup(states):
        i = cuda.grid(1)
        curand_init(1234, i, 0, states[i])

    # Random sampling kernel - computes the fraction of numbers with low bits set
    # from a random distribution.
    @cuda.jit(link=get_shims(), extensions=states_arg_handlers)
    def generate_kernel(states, sample_count, results):
        i = cuda.grid(1)
        count = 0

        # XXX: Copy state to local memory

        # Generate pseudo-random numbers
        for sample in range(sample_count):
            x = curand(states[i])

            # Check if low bit set
            if x & 1:
                count += 1

        # XXX: Copy state back to global memory

        # Store results
        results[i] += count

    # Create state on the device. The CUDA Array Interface provides a convenient
    # way to get the pointer needed for the shim functions.

    # Initialize cuRAND state

    states = States(nthreads)
    setup[blocks, threads](states)

    # Run random sampling kernel

    results = cuda.to_device(np.zeros(nthreads, dtype=np.int32))

    for i in range(repetitions):
        generate_kernel[blocks, threads](states, sample_count, results)

    # Collect the results and summarize them. This could have been done on
    # device, but the corresponding CUDA C++ sample does it on the host, and
    # we're following that example.

    host_results = results.copy_to_host()

    total = 0
    for i in range(nthreads):
        total += host_results[i]

    # A random distribution should have a fraction of 0.5 of low bits set
    fraction = np.float32(total) / np.float32(
        nthreads * sample_count * repetitions
    )

    assert np.isclose(fraction, 0.5, atol=1e-5)


@pytest.mark.parametrize(
    "States",
    [curandStatesXORWOW, curandStatesMRG32k3a, curandStatesPhilox4_32_10],
)
def test_curand_uniform(States):
    # State initialization kernel
    @cuda.jit(link=get_shims(), extensions=states_arg_handlers)
    def setup(states):
        i = cuda.grid(1)
        curand_init(1234, i, 0, states[i])

    @cuda.jit(link=get_shims(), extensions=states_arg_handlers)
    def count_upper_half(states, n, result):
        i = cuda.grid(1)
        count = 0

        # XXX: Copy state to local memory

        # Count the number of samples that falls greater than 0.5
        for sample in range(n):
            x = curand_uniform(states[i])
            if x > 0.5:
                count += 1

        # XXX: Copy state back to global memory

        result[i] += count

    states = States(nthreads)
    setup[blocks, threads](states)

    results = cuda.to_device(np.zeros(nthreads, dtype=np.int32))

    for i in range(repetitions):
        count_upper_half[blocks, threads](states, sample_count, results)

    host_results = results.copy_to_host()

    total = 0
    for i in range(nthreads):
        total += host_results[i]

    fraction = np.float32(total) / np.float32(
        sample_count * nthreads * repetitions
    )

    assert np.isclose(fraction, 0.5, atol=1e-4)


@pytest.mark.parametrize(
    "States",
    [curandStatesXORWOW, curandStatesMRG32k3a, curandStatesPhilox4_32_10],
)
def test_curand_normal(States):
    # State initialization kernel
    @cuda.jit(link=get_shims(), extensions=states_arg_handlers)
    def setup(states):
        i = cuda.grid(1)
        curand_init(1234, i, 0, states[i])

    @cuda.jit(link=get_shims(), extensions=states_arg_handlers)
    def count_within_1_std(states, n, result):
        i = cuda.grid(1)
        count = 0

        # XXX: Copy state to local memory

        # Count the number of samples that falls within 1 std from mean
        for sample in range(n):
            x = curand_normal(states[i])
            if -1.0 < x < 1.0:
                count += 1

        # XXX: Copy state back to global memory

        result[i] += count

    states = States(nthreads)
    setup[blocks, threads](states)

    results = cuda.to_device(np.zeros(nthreads, dtype=np.int32))

    for i in range(repetitions):
        count_within_1_std[blocks, threads](states, sample_count, results)

    host_results = results.copy_to_host()

    total = 0
    for i in range(nthreads):
        total += host_results[i]

    fraction = np.float32(total) / np.float32(
        sample_count * nthreads * repetitions
    )

    assert np.isclose(fraction, 0.682689492, atol=1e-4)


def test_curand_sobol_scramble():
    #  This program uses the device CURAND API to calculate what
    #  proportion of quasi-random 3D points fall within a sphere
    #  of radius 1, and to derive the volume of the sphere.
    #
    #  In particular it uses 64 bit scrambled Sobol direction
    #  vectors returned by curandGetDirectionVectors64, to
    #  generate double precision uniform samples.

    ffi = cffi.FFI()
    vector_size = 64

    @cuda.jit(link=get_shims(), extensions=states_arg_handlers)
    def setup(sobolDirectionVectors, sobolScrambleConstants, states):
        id = cuda.grid(1)
        dim = 3 * id

        for z in range(3):
            dirptr = ffi.from_buffer(
                sobolDirectionVectors[vector_size * (dim + z) :]
            )
            curand_init(
                dirptr,
                sobolScrambleConstants[dim + z],
                1234,
                states[dim + z],
            )

    @cuda.jit(link=get_shims(), extensions=states_arg_handlers)
    def count_within_unit_sphere(states, n, result):
        id = cuda.grid(1)
        baseDim = 3 * id
        count = 0

        # XXX: Copy state to local memory

        for sample in range(n):
            x = curand_uniform_double(states[baseDim])
            y = curand_uniform_double(states[baseDim + 1])
            z = curand_uniform_double(states[baseDim + 2])

            if x * x + y * y + z * z < 1.0:
                count += 1

        # XXX: Copy state back to global memory

        result[id] += count

    hostVectors = curandGetDirectionVectors64(
        nthreads * 3,
        curandDirectionVectorSet.CURAND_SCRAMBLED_DIRECTION_VECTORS_64_JOEKUO6,
    )
    sobolDirectionVectors = cuda.to_device(hostVectors)

    hostScrambleConstants = curandGetScrambleConstants64(nthreads * 3)
    sobolScrambleConstants = cuda.to_device(hostScrambleConstants)

    states = curandStatesScrambledSobol64(3 * nthreads)

    devResult = cuda.to_device(np.zeros(nthreads, dtype=np.int32))

    setup[blocks, threads](
        sobolDirectionVectors, sobolScrambleConstants, states
    )

    for i in range(repetitions):
        count_within_unit_sphere[blocks, threads](
            states, sample_count, devResult
        )

    result = devResult.copy_to_host()

    total = 0
    for i in range(nthreads):
        total += result[i]

    fraction = np.float64(total) / np.float64(
        sample_count * nthreads * repetitions
    )

    assert_allclose(fraction * 8.0, 4.0 / 3 * np.pi, atol=1e-4)


def test_curand_poisson_simple():
    HOURS = 16
    cashiers_load = (0, 1, 2, 3, 2, 1, 0, 1, 2, 3, 2, 1, 1, 1, 1, 1)

    @cuda.jit(link=get_shims(), extensions=states_arg_handlers)
    def setup(states):
        i = cuda.grid(1)
        curand_init(1234, i, 0, states[i])

    @cuda.jit(device=True)
    def update_queue(id, min, new_customers, queue_length, queue):
        balance = new_customers - 2 * cashiers_load[(min - 1) // 60]

        if balance + queue_length <= 0:
            queue_length = 0
        else:
            queue_length += balance

        queue[min - 1, id] = queue_length

        return queue_length

    @cuda.jit(link=get_shims(), extensions=states_arg_handlers)
    def simple_device_API_kernel(states, queue):
        id = cuda.grid(1)

        # XXX: Copy state to local memory

        queue_length = 0

        for min in range(1, 60 * HOURS + 1):
            new_customers = curand_poisson(
                states[id], 4 * (math.sin(min / 100.0) + 1)
            )
            queue_length = update_queue(
                id, min, new_customers, queue_length, queue
            )

    states = curandStatesXORWOW(nthreads)
    devResults = cuda.to_device(
        np.zeros((60 * HOURS, nthreads), dtype=np.uint32)
    )

    setup[blocks, threads](states)
    simple_device_API_kernel[blocks, threads](states, devResults)

    hostResults = devResults.copy_to_host()

    files = sorted(
        glob.glob(
            os.path.join(os.path.dirname(__file__), "simple_poisson_gold_*")
        )
    )

    gold = None
    for f in files:
        partial = np.load(f)
        if gold is None:
            gold = partial
        else:
            gold = np.concatenate((gold, partial), axis=0)

    np.testing.assert_equal(hostResults, gold)
