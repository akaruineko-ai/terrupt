"""Optional progress bars.

Wraps ``tqdm.auto`` so long-running generation and loading steps show live
progress. Degrades to a no-op when tqdm is not installed, keeping the library
dependency-free.
"""

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover
    _tqdm = None


class Progress:
    """Minimal tqdm wrapper supporting both library and CLI use."""

    def __init__(self, total=None, desc=None, unit="it", disable=False):
        self._total = total
        self._n = 0
        self._pbar = None
        if _tqdm is not None and not disable:
            self._pbar = _tqdm(total=total, desc=desc, unit=unit)

    def update(self, n=1):
        self._n += n
        if self._pbar is not None:
            self._pbar.update(n)

    def set_description(self, desc):
        if self._pbar is not None:
            self._pbar.set_description(desc)

    def close(self):
        if self._pbar is not None:
            self._pbar.close()
            self._pbar = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False