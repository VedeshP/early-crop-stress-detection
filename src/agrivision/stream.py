"""Readers for the 21 GB archive: a resumable HTTP stream or a local file.

The HTTP reader reconnects with a Range request from the current byte if the
connection drops, so a long Kaggle download survives network hiccups.
"""

import http.client
import io
import time
import urllib.request

DEFAULT_URL = (
    "https://intelinair-data-releases.s3.amazonaws.com/"
    "agriculture-vision/cvpr_challenge_2021/supervised/Agriculture-Vision-2021.tar.gz"
)


def s3_to_https(uri):
    if uri.startswith("s3://"):
        bucket, _, key = uri[5:].partition("/")
        return f"https://{bucket}.s3.amazonaws.com/{key}"
    return uri


class _Progress:
    def __init__(self, label, total, log, every=30.0):
        self.label, self.total, self.log, self.every = label, total, log, every
        self.start = self.last = time.time()

    def update(self, pos, force=False):
        now = time.time()
        if not force and now - self.last < self.every:
            return
        self.last = now
        rate = pos / max(now - self.start, 1e-6)
        msg = f"[{self.label}] {pos / 1e9:6.2f} GB"
        if self.total:
            eta = (self.total - pos) / max(rate, 1)
            msg += f" / {self.total / 1e9:.2f} GB ({100 * pos / self.total:5.1f}%)  ETA {eta / 60:5.1f} min"
        self.log(msg + f"  {rate / 1e6:6.1f} MB/s")


class ResumableHTTPReader(io.RawIOBase):
    def __init__(self, url, limit_bytes=0, retries=30, timeout=120, log=print, label="download"):
        self.url = s3_to_https(url)
        self.limit = int(limit_bytes or 0)
        self.retries, self.timeout, self.log = retries, timeout, log
        self.pos = 0
        self.total = None
        self.resp = None
        self._connect()
        self.progress = _Progress(label, min(self.total or 0, self.limit) if self.limit else self.total, log)

    def _connect(self):
        headers = {"User-Agent": "agrivision-toolkit"}
        if self.pos:
            headers["Range"] = f"bytes={self.pos}-"
        resp = urllib.request.urlopen(urllib.request.Request(self.url, headers=headers), timeout=self.timeout)
        if self.pos and resp.status != 206:
            raise IOError("server ignored the Range header; cannot resume")
        if self.total is None:
            length = resp.headers.get("Content-Length")
            self.total = int(length) if length else None
        self.resp = resp

    def readable(self):
        return True

    def readinto(self, b):
        if self.limit and self.pos >= self.limit:
            return 0
        view = memoryview(b)
        if self.limit:
            view = view[: self.limit - self.pos]
        for attempt in range(self.retries):
            try:
                n = self.resp.readinto(view)
                if n == 0 and self.total and self.pos < self.total:
                    raise IOError("connection closed early")
                self.pos += n
                self.progress.update(self.pos)
                return n
            except (OSError, http.client.HTTPException) as e:
                wait = min(60, 2 ** attempt)
                self.log(f"[download] {type(e).__name__}: {e} -> reconnecting at byte {self.pos} in {wait}s")
                time.sleep(wait)
                try:
                    self.resp.close()
                except Exception:
                    pass
                try:
                    self._connect()
                except (OSError, http.client.HTTPException) as e2:
                    self.log(f"[download] reconnect failed: {e2}")
        raise IOError(f"download failed after {self.retries} retries at byte {self.pos}")

    def close(self):
        if self.resp is not None:
            self.progress.update(self.pos, force=True)
            self.resp.close()
            self.resp = None
        super().close()


class FileReader(io.RawIOBase):
    def __init__(self, path, limit_bytes=0, log=print, label="read"):
        import os
        self.f = open(path, "rb")
        self.limit = int(limit_bytes or 0)
        self.pos = 0
        total = os.path.getsize(path)
        self.progress = _Progress(label, min(total, self.limit) if self.limit else total, log)

    def readable(self):
        return True

    def readinto(self, b):
        view = memoryview(b)
        if self.limit:
            if self.pos >= self.limit:
                return 0
            view = view[: self.limit - self.pos]
        n = self.f.readinto(view)
        self.pos += n
        self.progress.update(self.pos)
        return n

    def close(self):
        self.progress.update(self.pos, force=True)
        self.f.close()
        super().close()


def open_source(source, limit_bytes=0, log=print, label="download"):
    """Return a buffered binary stream for a URL, s3:// URI or local path."""
    if source.startswith(("http://", "https://", "s3://")):
        raw = ResumableHTTPReader(source, limit_bytes=limit_bytes, log=log, label=label)
    else:
        raw = FileReader(source, limit_bytes=limit_bytes, log=log, label=label)
    return io.BufferedReader(raw, buffer_size=16 << 20)
