"""Helpers for writing Spark DataFrames without pulling large results to the driver."""

from __future__ import annotations

from typing import Iterable, Optional

from pyspark.sql import DataFrame


def spark_write_parquet(
    df: DataFrame,
    out_dir: str,
    cols: Optional[Iterable[str]] = None,
    *,
    mode: str = "overwrite",
    compression: str = "snappy",
) -> None:
    """
    Write selected columns to a Parquet dataset path using Spark (distributed).

    Avoids pandas/toPandas and Hadoop FS edge cases from older notebooks.
    """
    out = df.select(*cols) if cols else df
    writer = (
        out.write.mode(mode)
        .option("compression", compression)
    )
    writer.parquet(out_dir)
