"""Tests for tidycop_hotspots.grid module."""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from tidycop_hotspots.grid import (
    make_grid,
    make_hexgrid,
    aggregate_points,
    aggregate_from_df,
)


# Washington DC bounding box (small area for fast tests)
DC_BOUNDS = (-77.12, 38.80, -76.91, 38.99)


class TestMakeGrid:
    def test_returns_geodataframe(self):
        grid = make_grid(DC_BOUNDS, cell_size_m=1000)
        assert isinstance(grid, gpd.GeoDataFrame)

    def test_has_cell_id(self):
        grid = make_grid(DC_BOUNDS, cell_size_m=1000)
        assert "cell_id" in grid.columns

    def test_cell_ids_unique(self):
        grid = make_grid(DC_BOUNDS, cell_size_m=1000)
        assert grid["cell_id"].nunique() == len(grid)

    def test_covers_bounds(self):
        grid = make_grid(DC_BOUNDS, cell_size_m=500)
        total_bounds = grid.total_bounds
        # Grid should cover at least the requested bounds
        assert total_bounds[0] <= DC_BOUNDS[0]
        assert total_bounds[1] <= DC_BOUNDS[1]
        assert total_bounds[2] >= DC_BOUNDS[2]
        assert total_bounds[3] >= DC_BOUNDS[3]

    def test_crs_preserved(self):
        grid = make_grid(DC_BOUNDS, cell_size_m=500, crs="EPSG:4326")
        assert grid.crs.to_epsg() == 4326

    def test_larger_cells_fewer_rows(self):
        small = make_grid(DC_BOUNDS, cell_size_m=250)
        large = make_grid(DC_BOUNDS, cell_size_m=1000)
        assert len(large) < len(small)


class TestMakeHexgrid:
    def test_returns_geodataframe(self):
        grid = make_hexgrid(DC_BOUNDS, cell_size_m=1000)
        assert isinstance(grid, gpd.GeoDataFrame)

    def test_has_cell_id(self):
        grid = make_hexgrid(DC_BOUNDS, cell_size_m=1000)
        assert "cell_id" in grid.columns

    def test_cells_are_polygons(self):
        grid = make_hexgrid(DC_BOUNDS, cell_size_m=1000)
        assert all(geom.geom_type == "Polygon" for geom in grid.geometry)


class TestAggregatePoints:
    def test_counts_points_in_cells(self):
        grid = make_grid(DC_BOUNDS, cell_size_m=2000)
        # Create some points within DC
        points = gpd.GeoDataFrame(
            geometry=[
                Point(-77.03, 38.90),
                Point(-77.03, 38.91),
                Point(-77.03, 38.90),  # same cell as first
                Point(-76.95, 38.85),
            ],
            crs="EPSG:4326",
        )
        result = aggregate_points(grid, points)
        assert "crime_count" in result.columns
        assert result["crime_count"].sum() == 4

    def test_zero_filled(self):
        grid = make_grid(DC_BOUNDS, cell_size_m=2000)
        points = gpd.GeoDataFrame(
            geometry=[Point(-77.03, 38.90)],
            crs="EPSG:4326",
        )
        result = aggregate_points(grid, points)
        # Most cells should have 0
        assert (result["crime_count"] == 0).sum() > 0
        assert result["crime_count"].min() == 0

    def test_custom_count_col(self):
        grid = make_grid(DC_BOUNDS, cell_size_m=2000)
        points = gpd.GeoDataFrame(
            geometry=[Point(-77.03, 38.90)],
            crs="EPSG:4326",
        )
        result = aggregate_points(grid, points, count_col="n_incidents")
        assert "n_incidents" in result.columns


class TestAggregateFromDf:
    def test_from_dataframe(self):
        grid = make_grid(DC_BOUNDS, cell_size_m=2000)
        df = pd.DataFrame({
            "std_latitude": [38.90, 38.91, 38.85],
            "std_longitude": [-77.03, -77.03, -76.95],
        })
        result = aggregate_from_df(grid, df)
        assert "crime_count" in result.columns
        assert result["crime_count"].sum() == 3

    def test_custom_lat_lon_cols(self):
        grid = make_grid(DC_BOUNDS, cell_size_m=2000)
        df = pd.DataFrame({
            "lat": [38.90],
            "lon": [-77.03],
        })
        result = aggregate_from_df(grid, df, lat_col="lat", lon_col="lon")
        assert result["crime_count"].sum() == 1
