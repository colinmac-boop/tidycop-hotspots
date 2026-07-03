"""Tests for tidycop_hotspots.validate module."""

import numpy as np

from tidycop_hotspots.validate import (
    predictive_accuracy_index,
    hit_rate_curve,
    recapture_rate,
    model_report,
)


def _make_predictions_actuals(n=100):
    """Create test predictions that correlate with actuals."""
    rng = np.random.default_rng(42)
    actuals = rng.poisson(5, n).astype(float)
    # Predictions correlated but noisy
    predictions = actuals + rng.normal(0, 1, n)
    return predictions, actuals


class TestPAI:
    def test_perfect_prediction(self):
        # If predictions perfectly rank actuals, PAI should be high
        actuals = np.array([10, 8, 6, 4, 2, 0, 0, 0, 0, 0], dtype=float)
        predictions = np.array([10, 8, 6, 4, 2, 0, 0, 0, 0, 0], dtype=float)
        pai = predictive_accuracy_index(predictions, actuals, area_pct=0.2)
        assert pai > 1.0  # Better than random

    def test_random_prediction_near_one(self):
        rng = np.random.default_rng(42)
        actuals = rng.poisson(5, 1000).astype(float)
        predictions = rng.uniform(0, 10, 1000)
        pai = predictive_accuracy_index(predictions, actuals, area_pct=0.2)
        # Random should be near 1.0 (within some variance)
        assert 0.5 < pai < 2.0

    def test_returns_float(self):
        predictions, actuals = _make_predictions_actuals()
        pai = predictive_accuracy_index(predictions, actuals)
        assert isinstance(pai, float)


class TestHitRateCurve:
    def test_returns_dataframe(self):
        predictions, actuals = _make_predictions_actuals()
        curve = hit_rate_curve(predictions, actuals)
        assert "area_pct" in curve.columns
        assert "crime_pct" in curve.columns
        assert "pai" in curve.columns

    def test_monotonically_increasing_crime_pct(self):
        predictions, actuals = _make_predictions_actuals()
        curve = hit_rate_curve(predictions, actuals)
        assert curve["crime_pct"].is_monotonic_increasing

    def test_ends_at_100_pct(self):
        predictions, actuals = _make_predictions_actuals()
        curve = hit_rate_curve(predictions, actuals, steps=20)
        assert curve["area_pct"].iloc[-1] == 1.0
        assert abs(curve["crime_pct"].iloc[-1] - 1.0) < 0.01


class TestRecaptureRate:
    def test_perfect_overlap(self):
        predictions = np.array([10, 8, 6, 4, 2, 0, 0, 0, 0, 0], dtype=float)
        actuals = np.array([10, 8, 6, 4, 2, 0, 0, 0, 0, 0], dtype=float)
        rate = recapture_rate(predictions, actuals, top_pct=0.2)
        assert rate == 1.0

    def test_returns_float_between_0_and_1(self):
        predictions, actuals = _make_predictions_actuals()
        rate = recapture_rate(predictions, actuals, top_pct=0.2)
        assert 0.0 <= rate <= 1.0


class TestModelReport:
    def test_contains_expected_keys(self):
        predictions, actuals = _make_predictions_actuals()
        report = model_report(predictions, actuals)
        assert "r2" in report
        assert "mae" in report
        assert "rmse" in report
        # PAI dict with thresholds
        assert "pai" in report
        assert 0.05 in report["pai"]

    def test_reasonable_values(self):
        predictions, actuals = _make_predictions_actuals()
        report = model_report(predictions, actuals)
        assert report["mae"] >= 0
        assert report["rmse"] >= 0
