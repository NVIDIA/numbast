import os
import shutil
import subprocess
import sys

from jinja2 import Environment, FileSystemLoader

from click.testing import CliRunner
from numbast.tools.static_binding_generator import static_binding_generator


def make_cfg(template, data_cuh, config_folder):
    env = Environment(loader=FileSystemLoader(os.path.dirname(template)))
    template = env.get_template(os.path.basename(template))
    cfg_path = os.path.join(config_folder, "override_entrypoint.yml")
    with open(cfg_path, "w") as f:
        f.write(template.render({"data": data_cuh}))

    return cfg_path


def test_shim_include_override_additional_import(tmpdir):
    """Tests:
    1. Additional Import field actually adds custom import libs in binding
    2. Shim Include Override overrides the shim include line
    """
    root = tmpdir
    config_folder = root.mkdir("config")
    output_folder = root.mkdir("output")
    here = os.path.dirname(os.path.abspath(__file__))

    src_data = os.path.join(here, "data.cuh")
    target_data = os.path.join(output_folder, "data.cuh")
    shutil.copy(src_data, target_data)
    cfg_template = os.path.join(here, "config/", "shim_include_override.yml.j2")
    cfg_path = make_cfg(cfg_template, target_data, config_folder)

    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        ["--cfg-path", cfg_path, "--output-dir", output_folder, "-fmt", False],
    )

    assert result.exit_code == 0

    with open(os.path.join(output_folder, "data.py")) as f:
        bindings = f.readlines()

    os_is_imported = False
    for line in bindings:
        # check import
        if line.startswith("import"):
            os_is_imported = True
        # check shim include override
        if line.startswith("shim_include"):
            assert "os" in line
    assert os_is_imported

    test_kernel_src = """
from numba import cuda
from data import Foo, add
@cuda.jit
def kernel():
    foo = Foo()
    one = add(foo.x, 1)

kernel[1, 1]()
"""

    test_kernel = os.path.join(output_folder, "test.py")
    with open(test_kernel, "w") as f:
        f.write(test_kernel_src)

    with open(os.path.join(output_folder, "data.py")) as f:
        binding = f.read()
        print(binding)

    res = subprocess.run(
        [sys.executable, test_kernel],
        cwd=output_folder,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )

    assert res.returncode == 0, res.stdout.decode("utf-8")
