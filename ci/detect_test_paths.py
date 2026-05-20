#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os
import sys
from pathlib import Path


MLIR_PREFIX = "numbast/src/numbast/experimental/mlir/"

BOTH_TEST_PATHS = {
    "numbast/src/numbast/__init__.py",
    "numbast/src/numbast/tools/static_binding_generator.py",
    "numbast/src/numbast/tools/static_binding_generator.schema.yaml",
    "numbast/src/numbast/tools/tests/test_mlir_backend_routing.py",
    "numbast/src/numbast/__main__.py",
    "numbast/pyproject.toml",
}

BOTH_TEST_PREFIXES = (
    ".github/workflows/",
    "ci/",
)


def classify_paths(paths: list[str]) -> tuple[bool, bool]:
    run_numba_cuda_tests = False
    run_mlir_tests = False

    for raw_path in paths:
        path = raw_path.strip()
        if not path:
            continue

        if path.startswith(MLIR_PREFIX):
            run_mlir_tests = True
        elif path in BOTH_TEST_PATHS or path.startswith(BOTH_TEST_PREFIXES):
            run_numba_cuda_tests = True
            run_mlir_tests = True
        else:
            run_numba_cuda_tests = True

    if not run_numba_cuda_tests and not run_mlir_tests:
        run_numba_cuda_tests = True
        run_mlir_tests = True

    return run_numba_cuda_tests, run_mlir_tests


def _write_github_output(
    run_numba_cuda_tests: bool, run_mlir_tests: bool
) -> None:
    output_path = os.environ.get("GITHUB_OUTPUT")
    if not output_path:
        return

    with Path(output_path).open("a", encoding="utf-8") as output_file:
        output_file.write(
            f"run_numba_cuda_tests={str(run_numba_cuda_tests).lower()}\n"
        )
        output_file.write(f"run_mlir_tests={str(run_mlir_tests).lower()}\n")


def main() -> int:
    paths = sys.stdin.read().splitlines()
    run_numba_cuda_tests, run_mlir_tests = classify_paths(paths)

    print(f"run_numba_cuda_tests={str(run_numba_cuda_tests).lower()}")
    print(f"run_mlir_tests={str(run_mlir_tests).lower()}")
    _write_github_output(run_numba_cuda_tests, run_mlir_tests)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
