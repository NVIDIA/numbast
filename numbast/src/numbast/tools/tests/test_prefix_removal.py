# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from numba import cuda


def test_prefix_removal(run_in_isolated_folder, arch_str):
    """Test that API prefix removal works correctly for function names."""
    res = run_in_isolated_folder(
        "prefix_removal.yml.j2",
        "prefix_removal.cuh",
        {"arch_str": arch_str},
        load_symbols=True,
        ruff_format=False,
    )

    run_result = res["result"]
    symbols = res["symbols"]
    alls = symbols["__all__"]

    assert run_result.exit_code == 0

    # Verify that the function is exposed as "foo" (without the "prefix_" prefix)
    assert "foo" in alls, f"Expected 'foo' in __all__, got: {alls}"
    assert "Foo" in alls, f"Expected 'Foo' in __all__, got: {alls}"

    # Verify that the original name "prefix_foo" is NOT exposed
    assert "prefix_foo" not in alls, (
        f"Expected 'prefix_foo' NOT in __all__, got: {alls}"
    )
    assert "__internal__Foo" not in alls, (
        f"Expected '__internal__Foo' NOT in __all__, got: {alls}"
    )
    assert "Bar" in alls, f"Expected 'Bar' in __all__, got: {alls}"
    assert "__internal__Bar" not in alls, (
        f"Expected '__internal__Bar' NOT in __all__, got: {alls}"
    )

    foo = symbols["foo"]
    Foo = symbols["Foo"]
    Bar = symbols["Bar"]

    @cuda.jit
    def kernel():
        """
        Exercise generated bindings for `foo`, `Foo`, and `Bar` in a CUDA kernel.

        This kernel invokes the top-level function `foo` with example arguments, constructs a `Foo` instance and accesses its `get_x()` method and `x` attribute, and reads the `BAR_A` and `BAR_B` attributes from `Bar`. It is intended for use in tests that verify API symbol exposure and runtime linkage.
        """
        result = foo(1, 2)  # noqa: F841
        foo_obj = Foo()
        x = foo_obj.get_x()  # noqa: F841
        x2 = foo_obj.x  # noqa: F841

        bar = Bar.BAR_A  # noqa: F841
        bar2 = Bar.BAR_B  # noqa: F841

    kernel[1, 1]()
