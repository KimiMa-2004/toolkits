'''

Author: Qimin Ma

Date: 2026-04-09 10:15:45

LastEditTime: 2026-04-09 10:32:24

FilePath: /Toolkit/toolkits/ml_dataset/dl_dataset.py

Description: Streaming IterableDataset for Polars LazyFrame (chunked scan, no full collect).

Copyright (c) 2026 by Qimin Ma, All Rights Reserved.

'''

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import polars as pl
import torch
from torch.utils.data import IterableDataset, get_worker_info


def _lazy_schema_names(lf: pl.LazyFrame) -> list[str]:
    if hasattr(lf, "collect_schema"):
        return lf.collect_schema().names()
    return lf.head(0).collect().columns


def train_test_hash_split(
    lf: pl.LazyFrame,
    *,
    id_col: str = "id",
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[pl.LazyFrame, pl.LazyFrame]:
    """Deterministic lazy split on ``id_col.hash(seed) % 100`` (no full-table collect)."""
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio must be in (0, 1)")
    cut = int((1.0 - test_ratio) * 100)
    h = pl.col(id_col).hash(seed)
    return lf.filter((h % 100) < cut), lf.filter((h % 100) >= cut)


def _collect_slice(lf: pl.LazyFrame, offset: int, length: int) -> pl.DataFrame:
    sliced = lf.slice(offset, length)
    for kwargs in ({"engine": "streaming"}, {"streaming": True}, {}):
        try:
            return sliced.collect(**kwargs)
        except TypeError:
            continue
    return sliced.collect()


def _feature_matrix_f32(
    chunk: pl.DataFrame,
    feature_cols: list[str],
    *,
    abs_clip: float | None,
) -> np.ndarray:
    """``(n, f)`` float32 matrix; categoricals → category codes as float."""
    sub = chunk.select(feature_cols)
    casts: list[pl.Expr] = []
    for c in feature_cols:
        if sub.schema[c] == pl.Categorical:
            casts.append(pl.col(c).to_physical().cast(pl.Float64))
        else:
            casts.append(pl.col(c).cast(pl.Float64, strict=False))
    arr = sub.select(casts).to_numpy()
    arr = np.nan_to_num(arr, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if abs_clip is not None:
        np.clip(arr, -abs_clip, abs_clip, out=arr)
    return np.ascontiguousarray(arr, dtype=np.float32)


class SimpleDataset(IterableDataset):
    """Yields micro-batches from ``lf.slice`` windows; use ``DataLoader(..., batch_size=None)``."""

    def __init__(
        self,
        data: pl.LazyFrame,
        target_col: str,
        feature_cols: list[str] | None = None,
        exclude_cols: list[str] | None = None,
        chunk_size: int = 50_000,
        batch_size: int = 512,
        shuffle: bool = True,
        shuffle_seed: int = 42,
        transform: Callable[[torch.Tensor], torch.Tensor] | None = None,
        drop_last: bool = True,
        feature_abs_clip: float | None = 1e6,
    ) -> None:
        super().__init__()
        self.data = data
        self.target_col = target_col
        exclude_cols = exclude_cols or []
        if feature_cols is None:
            names = _lazy_schema_names(data)
            feature_cols = [c for c in names if c not in [target_col, *exclude_cols]]
        self.feature_cols = feature_cols
        self.chunk_size = chunk_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed
        self.transform = transform
        self.drop_last = drop_last
        self.feature_abs_clip = feature_abs_clip

    @property
    def input_dim(self) -> int:
        return len(self.feature_cols)

    def __iter__(self):
        w = get_worker_info()
        wid, n_workers = (0, 1) if w is None else (w.id, w.num_workers)
        offset = wid * self.chunk_size
        stride = max(1, n_workers) * self.chunk_size
        chunk_idx = 0

        while True:
            chunk = _collect_slice(self.data, offset, self.chunk_size)
            if chunk.height == 0:
                break

            if self.shuffle:
                chunk = chunk.sample(
                    fraction=1.0, shuffle=True, seed=self.shuffle_seed + chunk_idx
                )

            x = torch.as_tensor(
                _feature_matrix_f32(
                    chunk, self.feature_cols, abs_clip=self.feature_abs_clip
                )
            )
            y = torch.as_tensor(
                np.ascontiguousarray(
                    chunk.select(self.target_col).to_numpy().ravel(), dtype=np.int64
                )
            )

            n = x.shape[0]
            for i in range(0, n, self.batch_size):
                bx, by = x[i : i + self.batch_size], y[i : i + self.batch_size]
                if self.drop_last and bx.shape[0] < self.batch_size:
                    continue
                if self.transform is not None:
                    bx = self.transform(bx)
                yield bx, by

            offset += stride
            chunk_idx += 1


__all__ = ["SimpleDataset", "train_test_hash_split"]
