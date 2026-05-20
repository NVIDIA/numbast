# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import os
from typing import Union
from tempfile import NamedTemporaryFile

from numba_cuda_mlir import cuda


class ShimWriterBase:
    """Base class for shim function management.

    During the lowering phase, each shim function is generated per C++ function,
    then written to a file or kept in memory. This class manages the registered
    shim functions and ensures they are not written twice.

    For Python functors that need to be lowered, they are compiled into PTX files.
    Currently, each PTX is lowered into a separate file. This class also manages
    these generated PTX files.

    Each shim writer has a ``preceding_text`` section that allows pre-defined text
    to precede the shim functions. This is most often used to add include statements.

    A shim writer should define the following methods:

    - `write_to_shim`: Add a shim function to the managed set of functions.
    - `write_to_ptx_shim`: Add a PTX to the managed set of PTXes.
    - `links`: Return a function which, when invoked, yields an iterator over the
      container (file or memory) for shim functions and PTXes.
    """

    def __init__(self, preceding_text):
        self.shim_written = {}
        self.ptx_written = {}
        self.preceding_text = preceding_text

    def write_to_shim(self, content: str, id: str): ...

    def write_to_ptx_shim(self, ptx: str, id: str): ...

    def links(
        self,
    ) -> list[Union[str, cuda.CUSource, cuda.PTXSource]]: ...


class FileShimWriter(ShimWriterBase):
    """Write shim functions to files via file I/O.

    For all Numba bindings, the shim functions are combined and written to a single file.
    For each PTX, a separate file is created.
    """

    def __init__(self, preceding_text=""):
        super().__init__(preceding_text)

        self.file = NamedTemporaryFile(mode="a+t", suffix=".cu", delete=False)
        self.file_name = self.file.name
        self.file.write(preceding_text)

    def write_to_shim(self, content: str, id: str):
        """Write a shim function to the source file, keyed by ``id`` to avoid duplication."""
        if id not in self.shim_written:
            self.shim_written[id] = content
            with open(self.file_name, "a") as f:
                f.write(content)

    def write_to_ptx_shim(self, ptx: str, id: str):
        """Write PTX to files, each keyed by ``id`` to avoid duplication (one file per PTX)."""
        if id not in self.ptx_written:
            tmp = NamedTemporaryFile(mode="w+t", suffix=".ptx", delete=False)
            self.ptx_written[tmp.name] = ptx
            with open(tmp.name, "w") as fh:
                fh.write(ptx)

    def __del__(self):
        if os.path.exists(self.file_name):
            os.remove(self.file_name)
        for f in self.ptx_written.keys():
            if os.path.exists(f):
                os.remove(f)

    def links(
        self,
    ) -> list[Union[str, cuda.CUSource, cuda.PTXSource]]:
        """Return an iterator of file paths containing shim functions and PTXes.

        Usage: ``declare_device(..., link=[*shim_writer.links()])``
        """

        shim_files = [self.file_name, *self.ptx_written.keys()]
        return [file for file in shim_files if os.path.exists(file)]


class MemoryShimWriter(ShimWriterBase):
    """Manage shim functions and PTX in memory.

    For each Numba bindings, the shim functions are combined into a single cuda.CUSource object.
    For each PTX, a separate cuda.PTXSource object is created.
    """

    def __init__(self, preceding_text=""):
        super().__init__(preceding_text)

        # Keep a live, mutable source buffer so CUSource consumers that capture
        # this object early (e.g., at @cuda.jit decoration time) still see shims
        # appended later during specialization/lowering.
        self._shim_stream = io.StringIO()
        if preceding_text:
            self._shim_stream.write(preceding_text)
            if not preceding_text.endswith("\n"):
                self._shim_stream.write("\n")
        self._shim_source = cuda.CUSource(self._shim_stream)
        self.shim_ptxes = []

    def write_to_shim(self, content: str, id: str):
        """Write a shim function into the in-memory CUSource, keyed by ``id`` to avoid duplication."""
        if id not in self.shim_written:
            self.shim_written[id] = content
            self._shim_stream.write(content)
            if not content.endswith("\n"):
                self._shim_stream.write("\n")

    def write_to_ptx_shim(self, ptx: str, id: str):
        """Write PTX into PTXSource objects, each keyed by ``id`` to avoid duplication."""
        if id not in self.ptx_written:
            self.ptx_written[id] = ptx
            self.shim_ptxes.append(ptx)

    def links(
        self,
    ) -> list[Union[str, cuda.CUSource, cuda.PTXSource]]:
        """Return an iterator over memory objects containing shim functions and PTXes.

        Usage: ``declare_device(..., link=[*shim_writer.links()])``
        """

        shim_ptxes = [cuda.PTXSource(ptx.encode()) for ptx in self.shim_ptxes]

        return [self._shim_source, *shim_ptxes]


__all__ = ["FileShimWriter", "MemoryShimWriter"]
