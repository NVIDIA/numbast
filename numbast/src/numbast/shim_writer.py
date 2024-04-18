# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os
from typing import Callable, Iterator, Union
from abc import abstractmethod, ABC
from tempfile import NamedTemporaryFile

from numba import cuda


class ShimWriterBase(ABC):
    """The base class for shim function management.

    During lowering phase, each shim function is generated per C++ function, then the generated
    shim function is written to a file or memory. This class manages each of the registered shim
    function to make sure shim functions are not written twice.

    For python functors that needs to be lowered, they are compiled into PTX files. Currently,
    each PTX is lowered into a separate file. This class also manages these generated PTX files.

    Each shim writer class has a `preceding_text` section that allows the user to add pre-defined
    text preceding the shim functions. This is most often used to add include statements.

    A shim writer should define the following methods:

    - `write_to_shim`: Add a shim function to the managed set of functions.
    - `write_to_ptx_shim`: Add a PTX to the managed set of PTXes.
    - `links`: Return a *function*. When the function is invoked,
    -   return an iterator to the container (file / memory container)
        for the shim functions and PTXes.
    """

    def __init__(self, preceding_text):
        self.shim_written = {}
        self.ptx_written = {}
        self.preceding_text = preceding_text

    @abstractmethod
    def write_to_shim(self, content: str, id: str):
        pass

    @abstractmethod
    def write_to_ptx_shim(self, ptx: str, id: str):
        pass

    @abstractmethod
    def links(
        self,
    ) -> Callable[[], Iterator[Union[str, cuda.CUSource, cuda.PTXSource]]]:
        pass


class FileShimWriter(ShimWriterBase):
    """File shim writer class writes the shim functions to file via file I/O.

    For all Numba bindings, the shim functions are combined and written to a single file.
    For each PTX, a separate file is created.
    """

    def __init__(self, preceding_text=""):
        super().__init__(preceding_text)

        self.file = NamedTemporaryFile(mode="a+t", suffix=".cu", delete=False)
        self.file_name = self.file.name
        self.file.write(preceding_text)

    def write_to_shim(self, content: str, id: str):
        """Write a shim function to the source file, denoted by an id to avoid duplication."""
        if id not in self.shim_written:
            self.shim_written[id] = content
            with open(self.file_name, "a") as f:
                f.write(content)

    def write_to_ptx_shim(self, ptx: str, id: str):
        """Write ptxes to the files, each keyed by an id to avoid duplication, one file per ptx."""
        if id not in self.ptx_written:
            f = NamedTemporaryFile(mode="w+t", suffix=".ptx", delete=False)
            self.ptx_written[f.name] = ptx
            with open(f.name, "w") as f:
                f.write(ptx)

    def __del__(self):
        if os.path.exists(self.file_name):
            os.remove(self.file_name)
        for f in self.ptx_written.keys():
            if os.path.exists(f):
                os.remove(f)

    @property
    def links(self) -> Callable[[], Iterator[str]]:
        """Return an iterator to the file containing shim functions and PTXes."""

        def iter_shim_files() -> Iterator[str]:
            for file in [self.file_name, *self.ptx_written.keys()]:
                if os.path.exists(file):
                    yield file

        return iter_shim_files


class MemoryShimWriter(ShimWriterBase):
    """The helper class to manage shim functions, storing them in memory.

    This class can only be enabled with pynvjitlink >= 0.2.0. pynvjitlink 0.2.0 patches numba
    linker with numba.cuda.CUSource and numba.cuda.PTXSource, which then allows numba linker
    to load these sources from memory.

    For each Numba bindings, the shim functions are combined into a single cuda.CUSource object.
    For each PTX, a separate cuda.PTXSource object is created.
    """

    def __init__(self, preceding_text=""):
        super().__init__(preceding_text)

        self.shim_funcs = []
        self.shim_ptxes = []

    def write_to_shim(self, content: str, id: str):
        """Write a shim function to CUSource, denoted by an id to avoid duplication."""
        if id not in self.shim_written:
            self.shim_written[id] = content
            self.shim_funcs.append(content)

    def write_to_ptx_shim(self, ptx: str, id: str):
        """Write ptxes to PTXSource, each keyed by an id to avoid duplication, one file per ptx."""
        if id not in self.ptx_written:
            self.ptx_written[id] = ptx
            self.shim_ptxes.append(ptx)

    @property
    def links(self) -> Callable[[], Iterator[Union[cuda.CUSource, cuda.PTXSource]]]:
        """Return an iterator to the memory reference containing shim functions and PTXes."""

        def iter_shim_files() -> Iterator[Union[cuda.CUSource, cuda.PTXSource]]:
            shim_source = cuda.CUSource(
                "\n".join([self.preceding_text] + self.shim_funcs)
            )
            shim_ptxes = [cuda.PTXSource(ptx.encode()) for ptx in self.shim_ptxes]

            for src in [shim_source, *shim_ptxes]:
                yield src

        return iter_shim_files


__all__ = ["FileShimWriter", "MemoryShimWriter"]
