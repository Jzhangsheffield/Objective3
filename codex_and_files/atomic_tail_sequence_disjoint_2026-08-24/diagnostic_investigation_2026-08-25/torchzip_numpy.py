from __future__ import annotations

import io
import pickle
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


_DTYPES = {
    "FloatStorage": np.dtype("<f4"),
    "DoubleStorage": np.dtype("<f8"),
    "HalfStorage": np.dtype("<f2"),
    "LongStorage": np.dtype("<i8"),
    "IntStorage": np.dtype("<i4"),
    "ShortStorage": np.dtype("<i2"),
    "CharStorage": np.dtype("i1"),
    "ByteStorage": np.dtype("u1"),
    "BoolStorage": np.dtype("?"),
    "BFloat16Storage": np.dtype("<u2"),
}


@dataclass(frozen=True)
class _StorageType:
    name: str

    @property
    def dtype(self) -> np.dtype:
        if self.name not in _DTYPES:
            raise ValueError(f"Unsupported PyTorch storage type: {self.name}")
        return _DTYPES[self.name]


@dataclass
class _Storage:
    array: np.ndarray


def _rebuild_tensor(storage: _Storage, storage_offset: int, size: tuple[int, ...], stride: tuple[int, ...]) -> np.ndarray:
    base = storage.array[int(storage_offset):]
    if not size:
        return np.asarray(base[0]).reshape(())
    byte_strides = tuple(int(value) * base.dtype.itemsize for value in stride)
    return np.lib.stride_tricks.as_strided(base, shape=tuple(int(v) for v in size), strides=byte_strides)


def _rebuild_tensor_v2(
    storage: _Storage,
    storage_offset: int,
    size: tuple[int, ...],
    stride: tuple[int, ...],
    requires_grad: bool,
    backward_hooks: Any,
    metadata: Any = None,
) -> np.ndarray:
    del requires_grad, backward_hooks, metadata
    return _rebuild_tensor(storage, storage_offset, size, stride)


class _TorchZipUnpickler(pickle.Unpickler):
    def __init__(self, handle: io.BytesIO, archive: zipfile.ZipFile, prefix: str) -> None:
        super().__init__(handle)
        self.archive = archive
        self.prefix = prefix
        self.storage_cache: dict[tuple[str, str], _Storage] = {}

    def find_class(self, module: str, name: str) -> Any:
        if module == "torch._utils" and name == "_rebuild_tensor_v2":
            return _rebuild_tensor_v2
        if module == "torch._utils" and name == "_rebuild_tensor":
            return _rebuild_tensor
        if module == "torch" and name.endswith("Storage"):
            return _StorageType(name)
        return super().find_class(module, name)

    def persistent_load(self, persistent_id: Any) -> Any:
        if not isinstance(persistent_id, tuple) or persistent_id[0] != "storage":
            raise pickle.UnpicklingError(f"Unsupported persistent id: {persistent_id!r}")
        _, storage_type, key, location, numel = persistent_id[:5]
        del location
        if not isinstance(storage_type, _StorageType):
            raise pickle.UnpicklingError(f"Unexpected storage type: {storage_type!r}")
        cache_key = (storage_type.name, str(key))
        if cache_key not in self.storage_cache:
            raw = self.archive.read(f"{self.prefix}/data/{key}")
            array = np.frombuffer(raw, dtype=storage_type.dtype, count=int(numel))
            self.storage_cache[cache_key] = _Storage(array)
        return self.storage_cache[cache_key]


def load_torch_zip(path: str | Path) -> Any:
    """Read common torch.save ZIP checkpoints/caches without importing PyTorch."""
    source = Path(path)
    with zipfile.ZipFile(source) as archive:
        data_names = [name for name in archive.namelist() if name.endswith("/data.pkl")]
        if len(data_names) != 1:
            raise ValueError(f"Expected one data.pkl in {source}, found {data_names}")
        data_name = data_names[0]
        prefix = data_name.rsplit("/", 1)[0]
        payload = archive.read(data_name)
        return _TorchZipUnpickler(io.BytesIO(payload), archive, prefix).load()
