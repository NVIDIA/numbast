import os

from jinja2 import Environment, FileSystemLoader

from click.testing import CliRunner
from numbast.tools.static_binding_generator import static_binding_generator


def make_cfg(template, data_cuh, config_folder):
    env = Environment(loader=FileSystemLoader(os.path.dirname(template)))
    template = env.get_template(os.path.basename(template))
    cfg_path = os.path.join(config_folder, "cfg.yml")
    with open(cfg_path, "w") as f:
        f.write(template.render({"data": data_cuh}))

    return cfg_path


def test_repro_info(tmpdir):
    """Consider both the config and output folder are traversible in the same
    tree, this PR makes sure that reproducible info accurately reflect where
    the config file can be located as a relative path to the binding file.
    """
    root = tmpdir
    config_folder = root.mkdir("config")
    output_folder = root.mkdir("output")
    here = os.path.dirname(os.path.abspath(__file__))

    data = os.path.join(here, "data.cuh")
    cfg_template = os.path.join(here, "config/", "cfg.yml.j2")
    cfg_path = make_cfg(cfg_template, data, config_folder)

    runner = CliRunner()
    result = runner.invoke(
        static_binding_generator,
        [
            "--cfg-path",
            cfg_path,
            "--output-dir",
            output_folder,
        ],
    )

    assert result.exit_code == 0

    with open(os.path.join(output_folder, "data.py")) as f:
        bindings = f.readlines()

    expected_info = {
        "Ast_canopy version",
        "Numbast version",
        "Generation command",
        "Static binding generator parameters",
        "Config file path (relative to the path of the generated binding)",
    }

    # Check that all expected info are present within the generated binding in
    # the form of line comments.
    for line in bindings:
        if not expected_info:
            break

        if line.startswith("#"):
            comment = line[1:].strip()
            if ":" in comment:
                keys = comment.split(":")
                for k in keys:
                    expected_info.discard(k)

    assert len(expected_info) == 0
