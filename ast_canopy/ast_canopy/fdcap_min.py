# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from enum import IntEnum
from tempfile import TemporaryFile
import os
from contextlib import contextmanager


class STREAMFD(IntEnum):
    STDIN = 0
    STDOUT = 1
    STDERR = 2


class MinFDCapture:
    """Capture a file descriptor by redirecting it to a temporary file.

    This is a minimal implementation and may miss corner cases. For example,
    stdin redirection is not currently supported.
    """

    def __init__(self, fd: STREAMFD):
        if fd == STREAMFD.STDIN:
            raise NotImplementedError("Cannot capture stdin")

        self.fd = fd
        self.old_fd = os.dup(fd)
        self.tmpfile = TemporaryFile()
        self.tmpfile_fd = self.tmpfile.fileno()

    def start(self):
        """Start capturing by redirecting the file descriptor to a temporary file."""
        try:
            os.fstat(self.old_fd)
        except (AttributeError, OSError) as e:
            raise ValueError("Old file descriptor is invalid") from e
        finally:
            os.dup2(self.tmpfile_fd, self.fd)

    def stop(self):
        """Restore the file descriptor using the saved descriptor."""
        os.dup2(self.old_fd, self.fd)

    def snap(self) -> str:
        """Return the captured content as a string."""
        self.tmpfile.seek(0)
        return self.tmpfile.read().decode("UTF-8")


@contextmanager
def capture_fd(fd: STREAMFD):
    """Context manager that captures a file descriptor and restores it after the block."""
    capturer = MinFDCapture(fd)
    capturer.start()
    try:
        yield capturer
    finally:
        capturer.stop()
