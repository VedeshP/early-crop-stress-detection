"""Packed storage: many small files concatenated into a few large shard files.

Kaggle datasets and network drives handle a few large files far better than
hundreds of thousands of small ones. Each record is addressed by
(shard file name, byte offset, byte length), stored in index.csv.
"""

import os
from pathlib import Path


class ShardWriter:
    def __init__(self, directory, prefix, max_bytes=1 << 30):
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.prefix = prefix
        self.max_bytes = int(max_bytes)
        self.index = -1
        self.file = None
        self.pos = 0
        self.total = 0
        self._next_shard()

    def _next_shard(self):
        if self.file:
            self.file.close()
        self.index += 1
        self.name = f"{self.prefix}_{self.index:03d}.bin"
        self.file = open(self.dir / self.name, "wb")
        self.pos = 0

    def add(self, data):
        if self.pos and self.pos + len(data) > self.max_bytes:
            self._next_shard()
        offset = self.pos
        self.file.write(data)
        self.pos += len(data)
        self.total += len(data)
        return self.name, offset, len(data)

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
            if self.pos == 0:
                (self.dir / self.name).unlink(missing_ok=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class ShardReader:
    """Random-access reads; safe to pickle into DataLoader / process-pool workers."""

    def __init__(self, directory):
        self.dir = Path(directory)
        self._files = {}
        self._pid = os.getpid()

    def read(self, name, offset, length):
        if os.getpid() != self._pid:
            self._files, self._pid = {}, os.getpid()
        f = self._files.get(name)
        if f is None:
            f = self._files[name] = open(self.dir / name, "rb")
        f.seek(int(offset))
        return f.read(int(length))

    def close(self):
        for f in self._files.values():
            f.close()
        self._files = {}

    def __getstate__(self):
        return {"dir": self.dir}

    def __setstate__(self, state):
        self.__init__(state["dir"])
