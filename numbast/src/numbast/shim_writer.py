# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os


class ShimWriter:
    """The helper class to write shim functions to source file."""

    def __init__(self, file_name, preceding_text):
        self.file_name = file_name
        self.written = {}
        self.ptx_written = {}

        with open(file_name, "w") as f:
            f.write(preceding_text)

    def write_to_shim(self, content, id):
        """Write a shim function to the source file, denoted by an id to avoid duplication."""
        assert os.path.exists(self.file_name)

        if id not in self.written:
            self.written[id] = content
            with open(self.file_name, "a") as f:
                f.write(content)

    def write_to_ptx_shim(self, ptx, id):
        """Write ptxes to the files, each keyed by an id to avoid duplication, one file per ptx."""
        if id not in self.ptx_written:
            self.ptx_written[f"{id}.ptx"] = ptx
            with open(f"{id}.ptx", "w") as f:
                f.write(ptx)

    def links(self):
        """Return the list of files to link to during compilation."""

        def iter_shim_files():
            for file in [self.file_name, *self.ptx_written.keys()]:
                if os.path.exists(file):
                    yield file

        return iter_shim_files
