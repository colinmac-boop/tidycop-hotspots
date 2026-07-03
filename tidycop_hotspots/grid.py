"""Spatial aggregation utilities for crime data.

Builds regular grids (square and hexagonal) over a bounding box and
aggregates point observations into per-cell counts. All builders accept
inputs in an arbitrary CRS (default WGS84) and internally project to a
suitable metric CRS so that ``cell_size_m`` is a real-world size.

Public API:
    - :func:`make_grid`
    - :func:`make_hexgrid`
    - :func:`aggregate_points`
    - :func:`aggregate_from_df`
"""

from __future__ import annotations

from typing import Tuple, Union

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon, box

BoundsLike = Tuple[float, float, float, float]
CRSLike = Union[str, int, "pyproj.CRS"]  # noqa: F821  (pyproj optional in type stubs)


# ---------------------------------------------------------------------------
# CRS helpers
# ---------------------------------------------------------------------------

def _pick_metric_crs(bounds: BoundsLike, crs: CRSLike) -> "pyproj.CRS":  # noqa: F821
    """Choose a reasonable metric CRS for the given bounds.

    Uses the appropriate UTM zone based on the bounds centroid (converted to
    geographic coordinates when necessary). Falls back to Web Mercator
    (EPSG:3857) at extreme latitudes where UTM is undefined.

    Parameters
    ----------
    bounds:
        ``(minx, miny, maxx, maxy)`` in the CRS given by ``crs``.
    crs:
        CRS of ``bounds``.

    Returns
    -------
    pyproj.CRS
        A projected CRS whose linear unit is metres.
    """
    from pyproj import CRS, Transformer

    src = CRS.from_user_input(crs)
    minx, miny, maxx, maxy = bounds
    cx = (minx + maxx) / 2.0
    cy = (miny + maxy) / 2.0

    if src.is_geographic:
        lon, lat = cx, cy
    else:
        transformer = Transformer.from_crs(src, "EPSG:4326", always_xy=True)
        lon, lat = transformer.transform(cx, cy)

    if lat >= 84.0 or lat <= -80.0:
        return CRS.from_epsg(3857)

    zone = int((lon + 180.0) // 6.0) + 1
    epsg = 32600 + zone if lat >= 0 else 32700 + zone
    return CRS.from_epsg(epsg)


def _reproject_bounds(
    bounds: BoundsLike, src_crs: CRSLike, dst_crs: CRSLike
) -> BoundsLike:
    """Reproject a bounding box between CRSes.

    Densifies the box slightly by using the four corners so that the returned
    bounds are the axis-aligned envelope of the reprojected polygon.
    """
    from pyproj import Transformer

    transformer = Transformer.from_crs(src_crs, dst_crs, always_xy=True)
    minx, miny, maxx, maxy = bounds
    xs = [minx, maxx, maxx, minx]
    ys = [miny, miny, maxy, maxy]
    px, py = transformer.transform(xs, ys)
    return (min(px), min(py), max(px), max(py))


# ---------------------------------------------------------------------------
# Grid builders
# ---------------------------------------------------------------------------

def make_grid(
    bounds: BoundsLike,
    cell_size_m: float,
    crs: CRSLike = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Build a square grid covering ``bounds``.

    Parameters
    ----------
    bounds:
        ``(minx, miny, maxx, maxy)`` in ``crs``.
    cell_size_m:
        Cell edge length in metres.
    crs:
        CRS of ``bounds`` and of the returned GeoDataFrame. Defaults to
        WGS84 (``EPSG:4326``).

    Returns
    -------
    geopandas.GeoDataFrame
        Grid cells with columns ``cell_id`` (int) and ``geometry``
        (:class:`shapely.geometry.Polygon`). Output CRS matches ``crs``.
    """
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")

    metric_crs = _pick_metric_crs(bounds, crs)
    minx, miny, maxx, maxy = _reproject_bounds(bounds, crs, metric_crs)

    xs = np.arange(minx, maxx + cell_size_m, cell_size_m)
    ys = np.arange(miny, maxy + cell_size_m, cell_size_m)

    polys: list[Polygon] = []
    for x0 in xs[:-1]:
        for y0 in ys[:-1]:
            polys.append(box(x0, y0, x0 + cell_size_m, y0 + cell_size_m))

    gdf = gpd.GeoDataFrame(
        {"cell_id": np.arange(len(polys), dtype=np.int64)},
        geometry=polys,
        crs=metric_crs,
    )
    return gdf.to_crs(crs)


def make_hexgrid(
    bounds: BoundsLike,
    cell_size_m: float,
    crs: CRSLike = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """Build a flat-top hexagonal grid covering ``bounds``.

    Parameters
    ----------
    bounds:
        ``(minx, miny, maxx, maxy)`` in ``crs``.
    cell_size_m:
        Hex "size" — the distance from centre to a vertex (i.e., the
        circumradius) in metres. Flat-top hexagons have a width of
        ``2 * cell_size_m`` and a height of ``sqrt(3) * cell_size_m``.
    crs:
        CRS of ``bounds`` and of the returned GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        Hex cells with columns ``cell_id`` (int) and ``geometry``
        (:class:`shapely.geometry.Polygon`).
    """
    if cell_size_m <= 0:
        raise ValueError("cell_size_m must be positive")

    metric_crs = _pick_metric_crs(bounds, crs)
    minx, miny, maxx, maxy = _reproject_bounds(bounds, crs, metric_crs)

    # Flat-top hex geometry
    w = 2.0 * cell_size_m           # full width (vertex-to-vertex, x axis)
    h = np.sqrt(3.0) * cell_size_m  # full height (flat-to-flat, y axis)
    dx = 0.75 * w                   # horizontal centre spacing
    dy = h                          # vertical centre spacing

    # Pad by one cell so the polygon extents cover the bbox edges
    nx = int(np.ceil((maxx - minx) / dx)) + 2
    ny = int(np.ceil((maxy - miny) / dy)) + 2

    def _hex_polygon(cx: float, cy: float) -> Polygon:
        # 6 vertices, flat-top → first vertex at angle 0
        angles = np.deg2rad(np.array([0, 60, 120, 180, 240, 300], dtype=float))
        xs = cx + cell_size_m * np.cos(angles)
        ys = cy + cell_size_m * np.sin(angles)
        return Polygon(zip(xs, ys))

    polys: list[Polygon] = []
    envelope = box(minx, miny, maxx, maxy)
    for i in range(nx):
        cx = minx + i * dx
        y_off = (h / 2.0) if (i % 2 == 1) else 0.0
        for j in range(ny):
            cy = miny + j * dy + y_off
            poly = _hex_polygon(cx, cy)
            if poly.intersects(envelope):
                polys.append(poly)

    gdf = gpd.GeoDataFrame(
        {"cell_id": np.arange(len(polys), dtype=np.int64)},
        geometry=polys,
        crs=metric_crs,
    )
    return gdf.to_crs(crs)


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def aggregate_points(
    grid_gdf: gpd.GeoDataFrame,
    points_gdf: gpd.GeoDataFrame,
    count_col: str = "crime_count",
) -> gpd.GeoDataFrame:
    """Count points per grid cell via spatial join.

    Parameters
    ----------
    grid_gdf:
        Grid cells with a ``cell_id`` column and polygon geometry.
    points_gdf:
        Point observations. Reprojected to ``grid_gdf``'s CRS when
        necessary.
    count_col:
        Name of the count column to add to the returned frame. Empty
        cells receive ``0``.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of ``grid_gdf`` with ``count_col`` appended.
    """
    if "cell_id" not in grid_gdf.columns:
        raise ValueError("grid_gdf must contain a 'cell_id' column")
    if grid_gdf.crs is None:
        raise ValueError("grid_gdf must have a CRS set")

    if len(points_gdf) == 0:
        out = grid_gdf.copy()
        out[count_col] = 0
        return out

    if points_gdf.crs is None:
        raise ValueError("points_gdf must have a CRS set")

    pts = points_gdf.to_crs(grid_gdf.crs) if points_gdf.crs != grid_gdf.crs else points_gdf

    joined = gpd.sjoin(
        pts[["geometry"]],
        grid_gdf[["cell_id", "geometry"]],
        how="inner",
        predicate="within",
    )
    counts = (
        joined.groupby("cell_id").size().rename(count_col).astype(np.int64)
    )

    out = grid_gdf.merge(counts, how="left", left_on="cell_id", right_index=True)
    out[count_col] = out[count_col].fillna(0).astype(np.int64)
    return out


def aggregate_from_df(
    grid_gdf: gpd.GeoDataFrame,
    df: pd.DataFrame,
    lat_col: str = "std_latitude",
    lon_col: str = "std_longitude",
    count_col: str = "crime_count",
) -> gpd.GeoDataFrame:
    """Aggregate a lat/lon DataFrame into per-cell counts.

    Convenience wrapper that constructs a GeoDataFrame in ``EPSG:4326`` from
    the given latitude/longitude columns and dispatches to
    :func:`aggregate_points`.

    Parameters
    ----------
    grid_gdf:
        Grid produced by :func:`make_grid` or :func:`make_hexgrid`.
    df:
        Incident-level DataFrame with latitude/longitude columns.
    lat_col, lon_col:
        Column names in ``df``.
    count_col:
        Name of the count column to add to the returned frame.

    Returns
    -------
    geopandas.GeoDataFrame
        Copy of ``grid_gdf`` with ``count_col`` appended.
    """
    if lat_col not in df.columns or lon_col not in df.columns:
        raise KeyError(f"df must contain '{lat_col}' and '{lon_col}' columns")

    valid = df[[lat_col, lon_col]].dropna()
    if len(valid) == 0:
        out = grid_gdf.copy()
        out[count_col] = 0
        return out

    points = gpd.GeoDataFrame(
        {"__idx": np.arange(len(valid), dtype=np.int64)},
        geometry=gpd.points_from_xy(valid[lon_col].to_numpy(), valid[lat_col].to_numpy()),
        crs="EPSG:4326",
    )
    return aggregate_points(grid_gdf, points, count_col=count_col)
