"""Feature engineering for grid-based crime prediction.

Every builder returns something aligned to ``grid_gdf.index`` so features
can be combined by :func:`build_feature_matrix`. All spatial math is done
in a metric CRS derived from ``grid_gdf`` (via :mod:`tidycop_hotspots.grid`)
so bandwidths, radii, and distances are in metres.

Public API:
    - :func:`kernel_density`
    - :func:`distance_to_nearest`
    - :func:`count_within_radius`
    - :func:`lagged_crime_counts`
    - :func:`build_feature_matrix`
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import geopandas as gpd
from scipy.spatial import cKDTree
from scipy.stats import gaussian_kde

from .grid import _pick_metric_crs  # local helper reused for CRS choice


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _metric_crs_for(grid_gdf: gpd.GeoDataFrame):
    """Pick a metric CRS that suits ``grid_gdf``."""
    if grid_gdf.crs is None:
        raise ValueError("grid_gdf must have a CRS set")
    bounds = tuple(grid_gdf.total_bounds)
    return _pick_metric_crs(bounds, grid_gdf.crs)


def _centroid_xy_metric(grid_gdf: gpd.GeoDataFrame) -> tuple[np.ndarray, np.ndarray, Any]:
    """Return centroid X/Y arrays in a metric CRS, plus the CRS itself."""
    metric = _metric_crs_for(grid_gdf)
    proj = grid_gdf.to_crs(metric)
    cent = proj.geometry.centroid
    return cent.x.to_numpy(), cent.y.to_numpy(), metric


def _points_xy_metric(
    poi_gdf: gpd.GeoDataFrame, metric_crs: Any
) -> tuple[np.ndarray, np.ndarray]:
    """Return X/Y arrays for point-like geometries in ``metric_crs``."""
    if poi_gdf.crs is None:
        raise ValueError("poi_gdf must have a CRS set")
    proj = poi_gdf.to_crs(metric_crs)
    # Fall back to geometry centroid for anything non-point-like
    geom = proj.geometry
    xs = geom.x.to_numpy() if (geom.geom_type == "Point").all() else geom.centroid.x.to_numpy()
    ys = geom.y.to_numpy() if (geom.geom_type == "Point").all() else geom.centroid.y.to_numpy()
    return xs, ys


# ---------------------------------------------------------------------------
# Kernel density
# ---------------------------------------------------------------------------

def kernel_density(
    grid_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    bandwidth_m: float = 500.0,
    kernel: str = "gaussian",
) -> pd.Series:
    """Kernel density estimate evaluated at each grid cell centroid.

    Uses :class:`scipy.stats.gaussian_kde` under the hood. The bandwidth is
    supplied in metres via ``bandwidth_m``; the underlying KDE is fitted in
    the metric CRS chosen for ``grid_gdf``.

    Parameters
    ----------
    grid_gdf:
        Grid cells (must have a CRS).
    points_gdf:
        Point observations. Reprojected to the metric CRS internally.
    bandwidth_m:
        Kernel bandwidth in metres. Interpreted as the desired 1-D standard
        deviation of the Gaussian kernel; scipy's ``bw_method`` scales the
        sample covariance, so we set it to ``bandwidth_m / sample_std``.
    kernel:
        Kernel identifier. Only ``"gaussian"`` is supported today; other
        values raise :class:`NotImplementedError`.

    Returns
    -------
    pandas.Series
        KDE density at each cell centroid, indexed like ``grid_gdf.index``
        and named ``"kde"``. Returns zeros when ``points_gdf`` is empty.
    """
    if kernel != "gaussian":
        raise NotImplementedError(f"kernel={kernel!r} not supported; use 'gaussian'")
    if bandwidth_m <= 0:
        raise ValueError("bandwidth_m must be positive")

    cx, cy, metric_crs = _centroid_xy_metric(grid_gdf)
    out_index = grid_gdf.index

    if len(points_gdf) == 0:
        return pd.Series(np.zeros(len(cx), dtype=float), index=out_index, name="kde")

    px, py = _points_xy_metric(points_gdf, metric_crs)
    if len(px) < 2 or np.allclose(px.std(), 0) or np.allclose(py.std(), 0):
        # gaussian_kde needs variability; fall back to zeros
        return pd.Series(np.zeros(len(cx), dtype=float), index=out_index, name="kde")

    sample = np.vstack([px, py])
    # Scale scipy's bw_method so the effective std is bandwidth_m.
    sample_std = float(np.sqrt(0.5 * (px.std(ddof=1) ** 2 + py.std(ddof=1) ** 2)))
    bw = bandwidth_m / sample_std if sample_std > 0 else 1.0
    kde = gaussian_kde(sample, bw_method=bw)
    values = kde(np.vstack([cx, cy]))
    return pd.Series(values, index=out_index, name="kde")


# ---------------------------------------------------------------------------
# Distance / neighbour features
# ---------------------------------------------------------------------------

def distance_to_nearest(
    grid_gdf: gpd.GeoDataFrame,
    poi_gdf: gpd.GeoDataFrame,
    col_name: str = "dist_nearest",
) -> pd.Series:
    """Distance (metres) from each cell centroid to the nearest POI.

    Parameters
    ----------
    grid_gdf:
        Grid cells.
    poi_gdf:
        Points of interest.
    col_name:
        Name of the returned Series.

    Returns
    -------
    pandas.Series
        Distance in metres, aligned to ``grid_gdf.index``. If ``poi_gdf`` is
        empty, every value is ``numpy.inf``.
    """
    cx, cy, metric_crs = _centroid_xy_metric(grid_gdf)

    if len(poi_gdf) == 0:
        return pd.Series(
            np.full(len(cx), np.inf, dtype=float),
            index=grid_gdf.index,
            name=col_name,
        )

    px, py = _points_xy_metric(poi_gdf, metric_crs)
    tree = cKDTree(np.column_stack([px, py]))
    dists, _ = tree.query(np.column_stack([cx, cy]), k=1)
    return pd.Series(dists.astype(float), index=grid_gdf.index, name=col_name)


def count_within_radius(
    grid_gdf: gpd.GeoDataFrame,
    poi_gdf: gpd.GeoDataFrame,
    radius_m: float = 500.0,
    col_name: str = "count_nearby",
) -> pd.Series:
    """Count POI points within ``radius_m`` metres of each cell centroid.

    Parameters
    ----------
    grid_gdf:
        Grid cells.
    poi_gdf:
        Points of interest.
    radius_m:
        Search radius in metres. Must be positive.
    col_name:
        Name of the returned Series.

    Returns
    -------
    pandas.Series
        Integer counts aligned to ``grid_gdf.index``.
    """
    if radius_m <= 0:
        raise ValueError("radius_m must be positive")

    cx, cy, metric_crs = _centroid_xy_metric(grid_gdf)
    out_index = grid_gdf.index

    if len(poi_gdf) == 0:
        return pd.Series(
            np.zeros(len(cx), dtype=np.int64), index=out_index, name=col_name
        )

    px, py = _points_xy_metric(poi_gdf, metric_crs)
    tree = cKDTree(np.column_stack([px, py]))
    neighbours = tree.query_ball_point(np.column_stack([cx, cy]), r=radius_m)
    counts = np.fromiter((len(ns) for ns in neighbours), dtype=np.int64, count=len(cx))
    return pd.Series(counts, index=out_index, name=col_name)


# ---------------------------------------------------------------------------
# Temporal lags
# ---------------------------------------------------------------------------

def lagged_crime_counts(
    grid_gdf: gpd.GeoDataFrame,
    incidents_df: pd.DataFrame,
    date_col: str = "std_incident_date",
    periods: Optional[List[Dict[str, Any]]] = None,
    cell_id_col: str = "cell_id",
    reference_date: Optional[pd.Timestamp] = None,
) -> pd.DataFrame:
    """Per-cell counts across trailing time windows.

    For each period ``{"name": "prev_month", "days": 30}`` the count of
    incidents in ``(reference_date - days, reference_date]`` is computed
    per ``cell_id_col``.

    Parameters
    ----------
    grid_gdf:
        Grid cells with a ``cell_id`` column matching ``incidents_df``.
    incidents_df:
        Incident-level DataFrame. Must contain ``date_col`` and
        ``cell_id_col``.
    date_col:
        Column of incident dates (any pandas-parseable dtype).
    periods:
        List of ``{"name": str, "days": int}`` dicts. Defaults to
        ``prev_month``/``prev_quarter``/``prev_year`` (30/90/365 days).
    cell_id_col:
        Cell identifier column present on both frames.
    reference_date:
        The "now" instant against which windows are measured. Defaults to
        the maximum date in ``incidents_df`` (or :func:`pandas.Timestamp.now`
        if the frame is empty).

    Returns
    -------
    pandas.DataFrame
        One column per period, indexed like ``grid_gdf.index`` and named
        after each period's ``name``.
    """
    if periods is None:
        periods = [
            {"name": "prev_month", "days": 30},
            {"name": "prev_quarter", "days": 90},
            {"name": "prev_year", "days": 365},
        ]

    if cell_id_col not in grid_gdf.columns:
        raise KeyError(f"grid_gdf missing '{cell_id_col}' column")

    cols = [p["name"] for p in periods]
    out = pd.DataFrame(
        0, index=grid_gdf.index, columns=cols, dtype=np.int64
    )

    if len(incidents_df) == 0:
        return out
    if cell_id_col not in incidents_df.columns or date_col not in incidents_df.columns:
        raise KeyError(
            f"incidents_df must contain '{cell_id_col}' and '{date_col}' columns"
        )

    inc = incidents_df[[cell_id_col, date_col]].copy()
    inc[date_col] = pd.to_datetime(inc[date_col], errors="coerce", utc=False)
    inc = inc.dropna(subset=[date_col, cell_id_col])
    if len(inc) == 0:
        return out

    ref = pd.to_datetime(reference_date) if reference_date is not None else inc[date_col].max()

    cell_to_row = pd.Series(grid_gdf.index, index=grid_gdf[cell_id_col].to_numpy())

    for period in periods:
        name = period["name"]
        days = int(period["days"])
        if days <= 0:
            raise ValueError(f"period '{name}' must have days > 0, got {days}")
        window_start = ref - pd.Timedelta(days=days)
        mask = (inc[date_col] > window_start) & (inc[date_col] <= ref)
        counts = inc.loc[mask].groupby(cell_id_col).size()
        if len(counts) == 0:
            continue
        rows = cell_to_row.reindex(counts.index).dropna().astype(np.int64)
        aligned = counts.reindex(rows.index)
        out.loc[rows.to_numpy(), name] = aligned.to_numpy(dtype=np.int64)

    return out


# ---------------------------------------------------------------------------
# Feature matrix assembly
# ---------------------------------------------------------------------------

def build_feature_matrix(
    grid_gdf: gpd.GeoDataFrame,
    feature_dict: Dict[str, Union[pd.Series, pd.DataFrame]],
) -> pd.DataFrame:
    """Combine per-cell feature Series/DataFrames into a single matrix.

    Every value is reindexed to ``grid_gdf.index``. Series become a single
    column named by the key (or the Series ``name`` if the key is empty);
    DataFrames are prefixed with ``"<key>__"`` when a key is provided.

    Parameters
    ----------
    grid_gdf:
        Reference grid; the returned frame's index matches ``grid_gdf.index``.
    feature_dict:
        Mapping of feature name → Series or DataFrame aligned (or
        alignable) to the grid index.

    Returns
    -------
    pandas.DataFrame
        Wide feature matrix. Missing values are left as NaN — callers are
        responsible for whatever imputation policy makes sense downstream.
    """
    frames: list[pd.DataFrame] = []
    for name, value in feature_dict.items():
        if isinstance(value, pd.Series):
            col_name = name or value.name or "feature"
            frames.append(value.reindex(grid_gdf.index).rename(col_name).to_frame())
        elif isinstance(value, pd.DataFrame):
            aligned = value.reindex(grid_gdf.index).copy()
            if name:
                aligned.columns = [f"{name}__{c}" for c in aligned.columns]
            frames.append(aligned)
        else:
            raise TypeError(
                f"feature_dict['{name}'] must be a Series or DataFrame, "
                f"got {type(value).__name__}"
            )

    if not frames:
        return pd.DataFrame(index=grid_gdf.index)

    return pd.concat(frames, axis=1)
