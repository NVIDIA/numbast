import click
import subprocess
import os

import pytest


MLIR_TESTS_DIR = "numbast/src/numbast/experimental/mlir"


def run_pytest(lib, test_dir, extra_pytest_args=None):
    if RAPIDS_TESTS_DIR := os.environ.get("RAPIDS_TESTS_DIR", None):
        junitxml = os.path.join(RAPIDS_TESTS_DIR, f"junit-{lib}.xml")
    else:
        junitxml = "/dev/null"

    extra_pytest_args = extra_pytest_args or []
    command = [
        "pytest",
        "-v",
        "-s",
        "--continue-on-collection-errors",
        "--cache-clear",
        f"--junitxml={junitxml}",
        *extra_pytest_args,
        *test_dir,
    ]
    try:
        subprocess.run(command, check=True, capture_output=False)
    except subprocess.CalledProcessError as e:
        if e.returncode not in {
            pytest.ExitCode.OK,
            pytest.ExitCode.NO_TESTS_COLLECTED,
        }:
            raise e


@click.command()
@click.option("--ast-canopy", is_flag=True, help="Run ast_canopy pytests.")
@click.option("--numbast", is_flag=True, help="Run numbast pytests.")
@click.option("--bf16", is_flag=True, help="Run bfloat16 pytests.")
@click.option("--cccl", is_flag=True, help="Run CCCL (CUB) binding pytests.")
@click.option("--all-tests", is_flag=True, help="Run all pytests.")
def run(
    ast_canopy: bool,
    numbast: bool,
    bf16: bool,
    cccl: bool,
    all_tests: bool,
):
    """Selectively run pytests in Numbast repo based on options provided.
    When `--all-tests` is specified, run all the tests. Otherwise, for
    each of the specified options, run tests corresponding to the specified
    package. `--all-tests` option is mutually exclusive to all other options.
    """
    if all_tests:
        if any([ast_canopy, numbast, bf16, cccl]):
            raise ValueError(
                "`all_tests` and any subpackage specs are mutual exclusive."
            )

    if all_tests or ast_canopy:
        run_pytest("ast_canopy", ["ast_canopy/"])
    if all_tests or numbast:
        run_pytest("numbast", ["numbast/"], [f"--ignore={MLIR_TESTS_DIR}"])
    if all_tests or bf16:
        run_pytest(
            "bf16",
            [
                "numbast_extensions/tests/test_bfloat16.py",
                "numbast_extensions/tests/test_bfloat162.py",
                "numbast_extensions/tests/static/test_static_bf16.py",
            ],
        )
    if all_tests or cccl:
        run_pytest("cccl", ["numbast_extensions/tests/thirdparty/CCCL/"])


if __name__ == "__main__":
    run()
