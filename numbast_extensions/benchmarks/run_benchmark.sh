#!/bin/bash

NUMBAST_BENCH_KERN_REPETITION=1000

BENCH_NAME=test_arithmetic_bf16

PY_NAME=${BENCH_NAME}.py
PY_PTX=${BENCH_NAME}_py.ptx

GOLD_NAME=${BENCH_NAME}_gold
GOLD_SRC_NAME=${GOLD_NAME}.cu
GOLD_PTX=${GOLD_NAME}.ptx

COMPUTE_CAP=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader|head -n 1)
SMCC=sm_${COMPUTE_CAP//./}

# Cleanup
rm -rf *.json *.nsys-rep *.sqlite $GOLD_NAME

# Compile gold
nvcc --gpu-architecture=$SMCC $GOLD_SRC_NAME -o $GOLD_NAME

# Prof gold
nsys profile --trace cuda --force-overwrite true -o gold.nsys-rep $GOLD_NAME

# Prof py LTO OFF
nsys profile --trace cuda --force-overwrite true -o py_lto_off.nsys-rep --env-var NUMBA_CUDA_ENABLE_PYNVJITLINK=1 python $PY_NAME --lto False

# Prof py LTO ON
nsys profile --trace cuda --force-overwrite true -o py_lto_on.nsys-rep --env-var NUMBA_CUDA_ENABLE_PYNVJITLINK=1 python $PY_NAME --lto True

# Create gold nsys stat report
nsys stats --report cuda_gpu_kern_sum --format json --output . gold.nsys-rep

# Analyze py LTO OFF nsys stat report
nsys stats --report cuda_gpu_kern_sum --format json --output . py_lto_off.nsys-rep

# Analyze py LTO ON nsys stat report
nsys stats --report cuda_gpu_kern_sum --format json --output . py_lto_on.nsys-rep

echo "Benchmark completes!"
echo "The below compares the performance between gold and Numba."
echo ""

# Compare stat report
python analyze.py gold_cuda_gpu_kern_sum.json py_lto_off_cuda_gpu_kern_sum.json py_lto_on_cuda_gpu_kern_sum.json
