"""Bounds-checked binary primitives."""
from __future__ import annotations
import struct
from pathlib import Path
import numpy as np
from .errors import BinaryFormatError


class BinaryReader:
    """Read a binary file with offset-rich truncation errors."""
    def __init__(self, path: str | Path, *, endianness: str = "little") -> None:
        self.path = Path(path)
        if endianness not in {"little", "big"}:
            raise ValueError(f"unsupported endianness {endianness!r}")
        try:
            self._data = self.path.read_bytes()
        except OSError as exc:
            raise BinaryFormatError(f"{self.path}: cannot read: {exc}") from exc
        self._pos = 0
        self._prefix = "<" if endianness == "little" else ">"

    def read_bytes(self, count: int) -> bytes:
        if count < 0:
            raise BinaryFormatError(f"{self.path} at offset {self._pos}: negative byte count {count}")
        end = self._pos + count
        if end > len(self._data):
            raise BinaryFormatError(f"{self.path} at offset {self._pos}: need {count} bytes, only {len(self._data)-self._pos} remain")
        value = self._data[self._pos:end]; self._pos = end
        return value

    def read_ascii(self, count: int) -> str:
        try: return self.read_bytes(count).decode("ascii")
        except UnicodeDecodeError as exc: raise BinaryFormatError(f"{self.path} at offset {self._pos-count}: invalid ASCII") from exc

    def _unpack(self, fmt: str): return struct.unpack(self._prefix + fmt, self.read_bytes(struct.calcsize(fmt)))[0]
    def read_int8(self) -> int: return self._unpack("b")
    def read_uint8(self) -> int: return self._unpack("B")
    def read_int32(self) -> int: return self._unpack("i")
    def read_uint32(self) -> int: return self._unpack("I")
    def read_int64(self) -> int: return self._unpack("q")
    def read_uint64(self) -> int: return self._unpack("Q")
    def read_float64(self) -> float: return self._unpack("d")
    def read_array(self, dtype, count: int) -> np.ndarray:
        dt = np.dtype(dtype).newbyteorder(self._prefix)
        if count < 0: raise BinaryFormatError(f"{self.path} at offset {self._pos}: negative array count")
        return np.frombuffer(self.read_bytes(dt.itemsize * count), dtype=dt, count=count).astype(dt.newbyteorder("="), copy=True)
    def read_length_prefixed_string(self) -> str:
        n = self.read_int32()
        if n < 0: raise BinaryFormatError(f"{self.path} at offset {self._pos-4}: negative string length {n}")
        return self.read_ascii(n)
    def tell(self) -> int: return self._pos
    def remaining_bytes(self) -> int: return len(self._data) - self._pos
    def expect_eof(self) -> None:
        if self.remaining_bytes(): raise BinaryFormatError(f"{self.path} at offset {self._pos}: {self.remaining_bytes()} unexpected trailing bytes")

