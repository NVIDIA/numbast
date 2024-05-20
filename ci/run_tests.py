import click
import subprocess


def run_pytest(lib, test_dir):
    command = [
        "pytest",
        "-v",
        "--continue-on-collection-errors",
        "--cache-clear",
        f'--junitxml="${{RAPIDS_TESTS_DIR}}/junit-{lib}.xml"',
        f"{test_dir}",
    ]
    subprocess.run(command)


@click.command()
@click.option("ast_canopy", type=bool)
@click.option("numbast", type=bool)
@click.option("bf16", type=bool)
@click.option("fp16", type=bool)
@click.option("curand_device", type=bool)
@click.option("all_tests", type=bool)
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
        run_pytest("ast_canopy", "ast_canopy/")
    if all_tests or numbast:
        run_pytest("numbast", "numbast/")
    if all_tests or bf16:
        run_pytest("bf16", "numba_extensions/tests/test_bf16*")
    if all_tests or fp16:
        run_pytest("fp16", "numba_extensions/tests/test_fp16*")
    if all_tests or curand_device:
        run_pytest("curand_device", "numba_extensions/tests/test_curand*")


if __name__ == "__main__":
    run()
