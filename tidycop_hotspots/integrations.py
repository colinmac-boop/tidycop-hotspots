"""Integration helpers for consuming tidycop DataFrames.

`tidycop` is an optional dependency — the helpers here operate on any
DataFrame that follows tidycop's ``std_*`` schema (``std_latitude``,
``std_longitude``, ``std_datetime``), so callers can either import
``tidycop`` themselves and pass the result of ``get_incidents(...)`` in,
or hand in any equivalently-shaped frame.

Public API:
    - :func:`from_tidycop` — one-call pipeline (df → grid + features + targets)
    - :class:`HotspotBundle` — dataclass holding the assembled artifacts
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd
import geopandas as gpd

from .grid import make_grid, make_hexgrid, aggregate_from_df
from .features import kernel_density, build_feature_matrix

BoundsLike = tuple[float, float, float, float]


@dataclass
class HotspotBundle:
    """Assembled artifacts from :func:`from_tidycop`.

    Attributes
    ----------
    grid:
        The grid GeoDataFrame with ``train_count`` and ``test_count``
        columns already attached.
    features:
        Feature matrix aligned to ``grid.index``.
    y_train:
        Training target Series (per-cell train count).
    y_test:
        Test target Series (per-cell test count), or ``None`` if
        ``train_end`` was not supplied.
    metadata:
        Dict describing the split: ``cutoff``, ``train_rows``,
        ``test_rows``, ``bounds``, ``cell_size_m``.
    """

    grid: gpd.GeoDataFrame
    features: pd.DataFrame
    y_train: pd.Series
    y_test: Optional[pd.Series]
    metadata: dict


def _infer_bounds(df: pd.DataFrame, lat_col: str, lon_col: str, pad: float) -> BoundsLike:
    valid = df[[lat_col, lon_col]].dropna()
    if len(valid) == 0:
        raise ValueError(
            "cannot infer bounds: no rows with non-null latitude/longitude"
        )
    minx = float(valid[lon_col].min()) - pad
    maxx = float(valid[lon_col].max()) + pad
    miny = float(valid[lat_col].min()) - pad
    maxy = float(valid[lat_col].max()) + pad
    return (minx, miny, maxx, maxy)


def from_tidycop(
    df: pd.DataFrame,
    *,
    train_end: Optional[Union[str, pd.Timestamp]] = None,
    bounds: Optional[BoundsLike] = None,
    cell_size_m: float = 250.0,
    grid_shape: str = "square",
    lat_col: str = "std_latitude",
    lon_col: str = "std_longitude",
    datetime_col: str = "std_datetime",
    bandwidth_m: float = 500.0,
    extra_points: Optional[dict[str, gpd.GeoDataFrame]] = None,
    bounds_pad: float = 0.005,
) -> HotspotBundle:
    """Turn a tidycop incident DataFrame into a ready-to-train bundle.

    This is the convenience seam between ``tidycop.get_incidents(...)`` and
    ``tidycop_hotspots``. The default column names match tidycop's ``std_*``
    schema. Any DataFrame with latitude/longitude columns will work.

    Pipeline
    --------
    1. Drop rows missing lat/lon.
    2. Split by ``train_end`` (inclusive of dates ≤ ``train_end`` in train).
       If ``train_end`` is ``None``, all rows become training and ``y_test``
       is ``None``.
    3. Build a square (or hex) grid over ``bounds`` (auto-inferred from the
       data when omitted, with a small ``bounds_pad`` in degrees).
    4. Aggregate train/test counts per cell.
    5. Compute a training-window KDE feature.
    6. Optionally add extra POI KDEs (``extra_points``: mapping of name →
       GeoDataFrame). Each is turned into a ``kde_<name>`` feature.

    Parameters
    ----------
    df:
        Incident-level DataFrame (typically ``tidycop.get_incidents(...)``).
    train_end:
        Cutoff date (inclusive) separating train/test. Anything ≤ this date
        goes into train; anything > goes into test. If ``None``, no split
        is performed.
    bounds:
        Explicit ``(minx, miny, maxx, maxy)`` in WGS84. Inferred from data
        when ``None``.
    cell_size_m:
        Grid cell edge (square) or circumradius (hex) in metres.
    grid_shape:
        ``"square"`` (default) or ``"hex"``.
    lat_col, lon_col, datetime_col:
        Column names on ``df``.
    bandwidth_m:
        Bandwidth for the training-KDE feature.
    extra_points:
        Optional mapping of feature name → GeoDataFrame of POIs. Each is
        added as a KDE column named ``kde_<name>``.
    bounds_pad:
        Degrees of padding around inferred bounds. Ignored when ``bounds``
        is explicit.

    Returns
    -------
    HotspotBundle
    """
    if grid_shape not in ("square", "hex"):
        raise ValueError(f"grid_shape must be 'square' or 'hex', got {grid_shape!r}")
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(
            f"df must contain '{lat_col}' and '{lon_col}' columns (got "
            f"{list(df.columns)[:10]}...)"
        )

    # 1. Drop missing lat/lon
    df_clean = df.dropna(subset=[lat_col, lon_col]).copy()
    if len(df_clean) == 0:
        raise ValueError("no rows with non-null latitude/longitude")

    # 2. Split
    if train_end is not None:
        if datetime_col not in df_clean.columns:
            raise KeyError(
                f"train_end supplied but df has no '{datetime_col}' column"
            )
        cutoff = pd.Timestamp(train_end)
        dt = pd.to_datetime(df_clean[datetime_col], errors="coerce", utc=True)
        # Normalise the cutoff to the same tz-awareness as the data
        if cutoff.tzinfo is None and dt.dt.tz is not None:
            cutoff = cutoff.tz_localize("UTC")
        train_df = df_clean[dt <= cutoff]
        test_df = df_clean[dt > cutoff]
    else:
        cutoff = None
        train_df = df_clean
        test_df = df_clean.iloc[0:0]  # empty

    # 3. Bounds + grid
    resolved_bounds = bounds or _infer_bounds(df_clean, lat_col, lon_col, bounds_pad)
    if grid_shape == "square":
        grid = make_grid(resolved_bounds, cell_size_m=cell_size_m)
    else:
        grid = make_hexgrid(resolved_bounds, cell_size_m=cell_size_m)

    # 4. Aggregate
    grid = aggregate_from_df(grid, train_df, lat_col=lat_col, lon_col=lon_col,
                             count_col="train_count")
    grid = aggregate_from_df(grid, test_df, lat_col=lat_col, lon_col=lon_col,
                             count_col="test_count")

    # 5. Training KDE feature
    features: dict[str, pd.Series] = {}
    if len(train_df) > 0:
        train_points = gpd.GeoDataFrame(
            geometry=gpd.points_from_xy(
                train_df[lon_col].to_numpy(),
                train_df[lat_col].to_numpy(),
            ),
            crs="EPSG:4326",
        )
        features["kde_train"] = kernel_density(
            grid, train_points, bandwidth_m=bandwidth_m
        )
    else:
        features["kde_train"] = pd.Series(0.0, index=grid.index, name="kde_train")

    # 6. Extra POI KDEs
    if extra_points:
        for name, poi_gdf in extra_points.items():
            if poi_gdf is None or len(poi_gdf) == 0:
                continue
            features[f"kde_{name}"] = kernel_density(
                grid, poi_gdf, bandwidth_m=bandwidth_m
            )

    feature_matrix = build_feature_matrix(grid, features)

    metadata = {
        "cutoff": str(cutoff) if cutoff is not None else None,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "bounds": tuple(float(v) for v in resolved_bounds),
        "cell_size_m": float(cell_size_m),
        "grid_shape": grid_shape,
        "bandwidth_m": float(bandwidth_m),
        "n_cells": int(len(grid)),
    }

    y_train = grid["train_count"].astype(float).rename("train_count")
    y_test = grid["test_count"].astype(float).rename("test_count") if cutoff is not None else None

    return HotspotBundle(
        grid=grid,
        features=feature_matrix,
        y_train=y_train,
        y_test=y_test,
        metadata=metadata,
    )
