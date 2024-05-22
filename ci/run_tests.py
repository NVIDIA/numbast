import click
import subprocess
import os


def run_pytest(lib, test_dir):
    if RAPIDS_TESTS_DIR := os.environ.get("RAPIDS_TESTS_DIR", None):
        junitxml = os.path.join(RAPIDS_TESTS_DIR, f"junit-{lib}.xml")
    else:
        junitxml = "/dev/null"

    command = [
        "pytest",
        "-v",
        "--continue-on-collection-errors",
        "--cache-clear",
        f"--junitxml={junitxml}",
        *test_dir,
    ]
    print(command)
    subprocess.run(command)


@click.command()
@click.option("--ast_canopy", is_flag=True)
@click.option("--numbast", is_flag=True)
@click.option("--bf16", is_flag=True)
@click.option("--fp16", is_flag=True)
@click.option("--curand_device", is_flag=True)
@click.option("--all_tests", is_flag=True)
def run(
    ast_canopy: bool,
    numbast: bool,
    bf16: bool,
    fp16: bool,
    curand_device: bool,
    all_tests: bool,
):
    if all_tests:
        if any([ast_canopy, numbast, bf16, fp16, curand_device]):
            raise ValueError(
                "`all_tests` and any subpackage specs are mutual exclusive."
            )

    if all_tests or ast_canopy:
        run_pytest("ast_canopy", ["ast_canopy/tests"])
    if all_tests or numbast:
        run_pytest("numbast", ["numbast/tests"])
    if all_tests or bf16:
        run_pytest(
            "bf16",
            [
                "numba_extensions/tests/test_bfloat16.py",
                "numba_extensions/tests/test_bfloat162.py",
            ],
        )
    if all_tests or fp16:
        run_pytest(
            "fp16",
            [
                "numba_extensions/tests/test_fp16.py",
                "numba_extensions/tests/test_fp162.py",
            ],
        )
    if all_tests or curand_device:
        run_pytest("curand_device", ["numba_extensions/tests/test_curand.py"])


if __name__ == "__main__":
    run()
