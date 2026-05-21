# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import subprocess
import sys
import textwrap


def test_experimental_import_does_not_load_numba_cuda_backend():
    script = textwrap.dedent(
        """
        import sys

        import numbast.experimental

        eager_modules = {
            "numbast.class_template",
            "numbast.enum",
            "numbast.function",
            "numbast.function_template",
            "numbast.shim_writer",
            "numbast.struct",
        }
        loaded = eager_modules.intersection(sys.modules)
        assert not loaded, sorted(loaded)
        assert numbast.__version__
        assert "bind_cxx_struct" in numbast.__all__
        """
    )
    subprocess.run([sys.executable, "-c", script], check=True)
