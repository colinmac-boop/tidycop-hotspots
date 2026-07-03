"""Tests for tidycop_hotspots.integrations.from_tidycop."""

import numpy as np
import pandas as pd
import geopandas as gpd
import pytest
from shapely.geometry import Point

import tidycop_hotspots as th
from tidycop_hotspots.integrations import from_tidycop, HotspotBundle


# Small Chicago-ish bounding box for realistic-ish coordinates
BOUNDS = (-87.72, 41.85, -87.58, 41.95)
SEED = 7


def _synthetic_df(n=800, with_dates=True):
    rng = np.random.default_rng(SEED)
    lons = rng.uniform(BOUNDS[0], BOUNDS[2], n)
    lats = rng.uniform(BOUNDS[1], BOUNDS[3], n)
    rows = {
        "std_latitude": lats,
        "std_longitude": lons,
        "std_offense": ["THEFT"] * n,
    }
    if with_dates:
        start = pd.Timestamp("2025-01-01", tz="UTC")
        offsets = rng.integers(0, 180, n)  # ~6 months
        rows["std_datetime"] = [start + pd.Timedelta(days=int(d)) for d in offsets]
    return pd.DataFrame(rows)


class TestFromTidycop:
    def test_returns_bundle(self):
        df = _synthetic_df()
        bundle = from_tidycop(df, cell_size_m=1000)
        assert isinstance(bundle, HotspotBundle)

    def test_grid_has_counts(self):
        df = _synthetic_df()
        bundle = from_tidycop(df, cell_size_m=1000)
        assert "train_count" in bundle.grid.columns
        assert "test_count" in bundle.grid.columns
        assert bundle.grid["train_count"].sum() > 0

    def test_no_split_when_train_end_missing(self):
        df = _synthetic_df()
        bundle = from_tidycop(df, cell_size_m=1000)
        # No cutoff → all rows train, y_test None
        assert bundle.y_test is None
        assert bundle.grid["test_count"].sum() == 0
        assert bundle.metadata["cutoff"] is None
        assert bundle.metadata["train_rows"] == len(df)

    def test_temporal_split(self):
        df = _synthetic_df()
        bundle = from_tidycop(
            df, train_end="2025-04-01", cell_size_m=1000
        )
        assert bundle.y_test is not None
        # Both should be non-empty for our 6-month synthetic window
        assert bundle.metadata["train_rows"] > 0
        assert bundle.metadata["test_rows"] > 0
        assert (
            bundle.metadata["train_rows"] + bundle.metadata["test_rows"]
            == len(df)
        )

    def test_feature_matrix_alignment(self):
        df = _synthetic_df()
        bundle = from_tidycop(df, cell_size_m=1000)
        assert len(bundle.features) == len(bundle.grid)
        assert (bundle.features.index == bundle.grid.index).all()
        assert "kde_train" in bundle.features.columns

    def test_explicit_bounds(self):
        df = _synthetic_df()
        bundle = from_tidycop(df, bounds=BOUNDS, cell_size_m=1000)
        assert bundle.metadata["bounds"] == tuple(float(v) for v in BOUNDS)

    def test_hex_grid_shape(self):
        df = _synthetic_df()
        bundle = from_tidycop(df, cell_size_m=1000, grid_shape="hex")
        # Hex cells have 6 vertices (7 with closing point)
        first = bundle.grid.geometry.iloc[0]
        assert len(first.exterior.coords) == 7

    def test_invalid_grid_shape(self):
        df = _synthetic_df()
        with pytest.raises(ValueError):
            from_tidycop(df, grid_shape="triangle")

    def test_missing_latlon_columns(self):
        df = pd.DataFrame({"lat": [1.0], "lon": [2.0]})
        with pytest.raises(KeyError):
            from_tidycop(df)

    def test_all_null_coords(self):
        df = pd.DataFrame({
            "std_latitude": [None, None],
            "std_longitude": [None, None],
        })
        with pytest.raises(ValueError):
            from_tidycop(df)

    def test_drops_null_coord_rows(self):
        df = _synthetic_df()
        # Null out a slice
        df.loc[:100, "std_latitude"] = np.nan
        bundle = from_tidycop(df, cell_size_m=1000)
        # Should still work, just with fewer rows counted
        assert bundle.metadata["train_rows"] == len(df) - 101

    def test_extra_points_kde(self):
        df = _synthetic_df()
        pois = gpd.GeoDataFrame(
            geometry=[
                Point(-87.65, 41.90),
                Point(-87.63, 41.88),
                Point(-87.68, 41.92),
            ],
            crs="EPSG:4326",
        )
        bundle = from_tidycop(
            df, cell_size_m=1000, extra_points={"bars": pois}
        )
        assert "kde_bars" in bundle.features.columns
        assert bundle.features["kde_bars"].notna().any()

    def test_metadata_content(self):
        df = _synthetic_df()
        bundle = from_tidycop(df, cell_size_m=500, train_end="2025-04-01")
        md = bundle.metadata
        assert md["cell_size_m"] == 500.0
        assert md["grid_shape"] == "square"
        assert md["bandwidth_m"] == 500.0
        assert md["n_cells"] == len(bundle.grid)

    def test_train_end_without_datetime_column(self):
        df = _synthetic_df(with_dates=False)
        with pytest.raises(KeyError):
            from_tidycop(df, train_end="2025-04-01")

    def test_naive_cutoff_with_tz_aware_data(self):
        # Regression test for tz handling
        df = _synthetic_df()
        # Should not raise
        bundle = from_tidycop(df, train_end="2025-03-15", cell_size_m=1000)
        assert bundle.y_test is not None
