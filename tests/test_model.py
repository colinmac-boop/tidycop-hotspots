"""Tests for tidycop_hotspots.model module."""

import numpy as np
import pandas as pd
import tempfile
from pathlib import Path

from tidycop_hotspots.model import (
    HotspotForest,
    HotspotClassifier,
    train_test_temporal_split,
)


def _make_training_data(n=200):
    """Create synthetic crime data for testing."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({
        "kde": rng.uniform(0, 1, n),
        "dist_transit": rng.uniform(0, 5000, n),
        "dist_bars": rng.uniform(0, 3000, n),
        "pop_density": rng.uniform(100, 10000, n),
    })
    # Target loosely correlated with features
    y = (X["kde"] * 10 + rng.poisson(2, n)).astype(float)
    return X, y


class TestHotspotForest:
    def test_fit_predict(self):
        X, y = _make_training_data()
        model = HotspotForest(n_estimators=50)
        model.fit(X, y)
        preds = model.predict(X)
        assert len(preds) == len(X)
        assert model.is_fitted

    def test_feature_importance(self):
        X, y = _make_training_data()
        model = HotspotForest(n_estimators=50)
        model.fit(X, y)
        imp = model.feature_importance()
        assert isinstance(imp, pd.DataFrame)
        assert "feature" in imp.columns
        assert "importance" in imp.columns
        assert len(imp) == X.shape[1]
        # Should be sorted descending
        assert imp["importance"].is_monotonic_decreasing

    def test_cross_validate(self):
        X, y = _make_training_data()
        model = HotspotForest(n_estimators=50)
        results = model.cross_validate(X, y, cv=3)
        assert "r2" in results
        assert "mae" in results
        assert "rmse" in results
        assert len(results["r2"]) == 3

    def test_save_load(self):
        X, y = _make_training_data()
        model = HotspotForest(n_estimators=50)
        model.fit(X, y, feature_names=list(X.columns))

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "model.joblib"
            model.save(path)
            loaded = HotspotForest.load(path)
            assert loaded.is_fitted
            assert loaded.feature_names_ == model.feature_names_
            np.testing.assert_array_almost_equal(
                loaded.predict(X), model.predict(X)
            )

    def test_feature_names_from_dataframe(self):
        X, y = _make_training_data()
        model = HotspotForest(n_estimators=50)
        model.fit(X, y)
        assert model.feature_names_ == list(X.columns)


class TestHotspotClassifier:
    def test_fit_predict(self):
        X, y = _make_training_data()
        model = HotspotClassifier(n_estimators=50, threshold=0.75)
        model.fit(X, y)
        preds = model.predict(X)
        assert set(np.unique(preds)).issubset({0, 1})

    def test_predict_proba(self):
        X, y = _make_training_data()
        model = HotspotClassifier(n_estimators=50, threshold=0.75)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert len(proba) == len(X)
        assert proba.min() >= 0
        assert proba.max() <= 1


class TestTemporalSplit:
    def test_splits_by_date(self):
        n = 100
        data = pd.DataFrame({
            "cell_id": range(n),
            "period": pd.date_range("2024-01-01", periods=n, freq="D"),
            "feat1": np.random.randn(n),
            "crime_count": np.random.poisson(3, n),
        })
        X_train, y_train, X_test, y_test = train_test_temporal_split(
            data,
            target_col="crime_count",
            feature_cols=["feat1"],
            train_end_date="2024-03-01",
            test_start_date="2024-03-15",
        )
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(X_train) + len(X_test) <= n
