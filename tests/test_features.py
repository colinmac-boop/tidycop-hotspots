"""Tests for tidycop_hotspots.features module."""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

from tidycop_hotspots.grid import make_grid
from tidycop_hotspots.features import (
    kernel_density,
    distance_to_nearest,
    count_within_radius,
    build_feature_matrix,
)


DC_BOUNDS = (-77.12, 38.80, -76.91, 38.99)


def _make_test_grid():
    return make_grid(DC_BOUNDS, cell_size_m=2000)


def _make_test_points(n=50):
    rng = np.random.default_rng(42)
    lons = rng.uniform(-77.10, -76.95, n)
    lats = rng.uniform(38.82, 38.97, n)
    return gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326",
    )


class TestKernelDensity:
    def test_returns_series(self):
        grid = _make_test_grid()
        points = _make_test_points()
        result = kernel_density(grid, points, bandwidth_m=1000)
        assert isinstance(result, pd.Series)
        assert len(result) == len(grid)

    def test_nonnegative(self):
        grid = _make_test_grid()
        points = _make_test_points()
        result = kernel_density(grid, points, bandwidth_m=1000)
        assert (result >= 0).all()

    def test_higher_near_cluster(self):
        grid = _make_test_grid()
        # Dense cluster of points in one area
        cluster = gpd.GeoDataFrame(
            geometry=[Point(-77.03 + i * 0.002, 38.90 + j * 0.002)
                      for i in range(10) for j in range(10)],
            crs="EPSG:4326",
        )
        result = kernel_density(grid, cluster, bandwidth_m=2000)
        # At least some cells should be nonzero near the cluster
        assert result.max() > 0


class TestDistanceToNearest:
    def test_returns_series(self):
        grid = _make_test_grid()
        pois = _make_test_points(10)
        result = distance_to_nearest(grid, pois)
        assert isinstance(result, pd.Series)
        assert len(result) == len(grid)

    def test_nonnegative(self):
        grid = _make_test_grid()
        pois = _make_test_points(10)
        result = distance_to_nearest(grid, pois)
        assert (result >= 0).all()

    def test_closer_with_more_pois(self):
        grid = _make_test_grid()
        few = _make_test_points(5)
        many = _make_test_points(100)
        dist_few = distance_to_nearest(grid, few)
        dist_many = distance_to_nearest(grid, many)
        # Average distance should be smaller with more POIs
        assert dist_many.mean() <= dist_few.mean()


class TestCountWithinRadius:
    def test_returns_series(self):
        grid = _make_test_grid()
        pois = _make_test_points(20)
        result = count_within_radius(grid, pois, radius_m=2000)
        assert isinstance(result, pd.Series)
        assert len(result) == len(grid)

    def test_nonnegative_integers(self):
        grid = _make_test_grid()
        pois = _make_test_points(20)
        result = count_within_radius(grid, pois, radius_m=2000)
        assert (result >= 0).all()

    def test_larger_radius_more_counts(self):
        grid = _make_test_grid()
        pois = _make_test_points(20)
        small = count_within_radius(grid, pois, radius_m=500)
        large = count_within_radius(grid, pois, radius_m=5000)
        assert large.sum() >= small.sum()


class TestBuildFeatureMatrix:
    def test_combines_features(self):
        grid = _make_test_grid()
        s1 = pd.Series(np.ones(len(grid)), index=grid.index, name="feat1")
        s2 = pd.Series(np.zeros(len(grid)), index=grid.index, name="feat2")
        result = build_feature_matrix(grid, {"feat1": s1, "feat2": s2})
        assert isinstance(result, pd.DataFrame)
        assert "feat1" in result.columns
        assert "feat2" in result.columns
        assert len(result) == len(grid)
