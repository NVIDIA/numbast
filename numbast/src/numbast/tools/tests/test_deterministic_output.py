# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generated bindings must not change from one run to the next.

Python randomises `str` hashing per interpreter (PYTHONHASHSEED), so anything
emitted by iterating a `set` lands in a different order every run. That makes
regenerating a binding produce a large, meaningless diff, which in turn breaks
build caching and makes it impossible to review a regeneration or to verify
that a refactor changed nothing.
"""

import os
import shutil
import subprocess
import sys


from jinja2 import Environment, FileSystemLoader

from numbast.static.renderer import BaseRenderer, get_rendered_imports

HERE = os.path.dirname(os.path.abspath(__file__))


# -- fast canary ---------------------------------------------------------


def test_import_block_is_emitted_in_a_canonical_order():
    """The import block is sorted, so its order cannot depend on set iteration.

    Asserting sortedness rather than comparing two runs keeps this
    deterministic: two sets holding the same strings are not *guaranteed* to
    iterate differently, so a two-run comparison could pass by luck.
    """
    BaseRenderer.Imports.clear()
    BaseRenderer.Imports.update(
        {
            "from numba.cuda.types import int32",
            "import io",
            "from numba import types",
            "import numba",
            "from numba.cuda.cudaimpl import lower",
            "from numba.cuda.extending import as_numba_type",
        }
    )
    try:
        lines = [
            line for line in get_rendered_imports().splitlines() if line.strip()
        ]
    finally:
        BaseRenderer.Imports.clear()

    assert lines == sorted(lines), lines


# -- the real guarantee --------------------------------------------------


def _make_project(tmp_path, arch_str):
    """One config + header tree, reused by every run.

    Regenerating *in place* is the property that matters, and it also keeps
    absolute paths -- which the binding embeds -- out of the comparison.
    """
    config_folder = tmp_path / "config"
    output_folder = tmp_path / "output"
    config_folder.mkdir(parents=True, exist_ok=True)
    output_folder.mkdir(parents=True, exist_ok=True)

    header = "data.cuh"
    shutil.copy(os.path.join(HERE, header), output_folder / header)

    env = Environment(loader=FileSystemLoader(HERE))
    template = env.get_template("config/cfg.yml.j2")
    config_path = config_folder / "cfg.yml"
    config_path.write_text(
        template.render(
            {"data": str(output_folder / header), "arch_str": arch_str}
        )
    )
    return config_path, output_folder


def _generate(project, hash_seed):
    """Regenerate the binding in a subprocess with a given PYTHONHASHSEED.

    A subprocess is required: PYTHONHASHSEED is read at interpreter start, so
    an in-process CliRunner cannot vary it and cannot observe this bug.
    """
    config_path, output_folder = project
    child_env = dict(os.environ, PYTHONHASHSEED=str(hash_seed))
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "numbast",
            "--cfg-path",
            str(config_path),
            "--output-dir",
            str(output_folder),
            "-fmt",
            "false",
        ],
        capture_output=True,
        text=True,
        env=child_env,
    )
    assert proc.returncode == 0, proc.stderr

    binding = output_folder / "data.py"
    assert binding.is_file(), proc.stdout
    return binding.read_text()


def test_generation_is_byte_identical_across_hash_seeds(tmp_path, arch_str):
    project = _make_project(tmp_path, arch_str)
    assert _generate(project, 1) == _generate(project, 999)


def test_generation_emits_the_same_lines_in_the_same_order(tmp_path, arch_str):
    """Guards the failure mode specifically: same lines, shuffled.

    Before the fix the two runs were identical as *multisets* of lines and
    differed only in order, so a test comparing sorted output would have
    passed while the bug was live. Checking the multiset first also means a
    failure says which kind of change happened.
    """
    project = _make_project(tmp_path, arch_str)
    first = _generate(project, 1).splitlines()
    second = _generate(project, 999).splitlines()

    assert sorted(first) == sorted(second), "the generated content changed"
    assert first == second, "same lines in a different order -- unstable sort"
