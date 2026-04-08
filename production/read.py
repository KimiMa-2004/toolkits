from __future__ import annotations

import os
from datetime import date, datetime

import duckdb
import pandas as pd
import polars as pl

from toolkits.production.production import _coerce_date

data_root = os.getenv("DATA_ROOT") if os.getenv("DATA_ROOT") is not None else "./data"

DateLike = date | datetime | str | pd.Timestamp


def read_constant(
    dir_name: str,
    file_name: str,
    mode: str = "pandas",
    table_name: str = "data",
) -> pd.DataFrame | pl.DataFrame | None:
    path = os.path.join(data_root, dir_name, file_name).replace("\\", "/")
    if mode == "pandas":
        return pd.read_parquet(path)
    if mode == "polars":
        return pl.read_parquet(path)
    if mode == "duckdb":
        duckdb.sql(
            f"create or replace table {table_name} as select * from read_parquet('{path}')"
        )
        return None
    raise ValueError(f"Invalid mode: {mode}")


def read_timeseries(
    dir_name: str,
    file_name: str,
    start: DateLike,
    end: DateLike,
    mode: str = "pandas",
    table_name: str = "data",
    date_col: str = "TradingDay",
) -> pd.DataFrame | pl.DataFrame | None:
    """按 ``[start, end]``（含端点）按**日历日**筛选；``start``/``end`` 与 ``production`` 一样支持 ``date`` / 字符串等。"""
    start_d = _coerce_date(start)
    end_d = _coerce_date(end)
    path = os.path.join(data_root, dir_name, file_name).replace("\\", "/")

    if mode == "pandas":
        data = pd.read_parquet(path)
        ts = pd.to_datetime(data[date_col])
        day = ts.dt.normalize()
        m = (day >= pd.Timestamp(start_d)) & (day <= pd.Timestamp(end_d))
        return data.loc[m]

    if mode == "polars":
        data = pl.read_parquet(path)
        col = pl.col(date_col)
        sch = data.schema[date_col]
        if sch == pl.Date:
            d = col
        else:
            d = col.cast(pl.Datetime).dt.date()
        return data.filter(
            (d >= pl.lit(start_d)) & (d <= pl.lit(end_d))
        )

    if mode == "duckdb":
        duckdb.sql(
            f"""
            create or replace table {table_name} as
            select * from read_parquet('{path}')
            where cast({date_col} as date) >= date '{start_d.isoformat()}'
              and cast({date_col} as date) <= date '{end_d.isoformat()}'
            """
        )
        return None

    raise ValueError(f"Invalid mode: {mode}")
