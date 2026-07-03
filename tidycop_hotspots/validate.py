"""Prediction evaluation metrics for crime forecasting.

The core metric here is the **Predictive Accuracy Index (PAI)** from
Chainey, Tompson & Uhlig (2008): the fraction of crime captured within
the top ``a%`` of predicted-hot area, divided by ``a``. A PAI of 1 means
predictions are no better than a uniform baseline; PAI of 10 means the
top 5% of cells captured 50% of crime, which is quite good.

Companion helpers here:

* :func:`hit_rate_curve` — the "how much crime do we catch at each
  coverage level" curve underlying most of these metrics.
* :func:`recapture_rate` — cell-level agreement between predicted and
  actual hotspots.
* :func:`surveillance_plot` — matplotlib rendering of the hit-rate curve
  vs. random baseline. Optional; ``matplotlib`` is imported lazily.
* :func:`model_report` — one-call summary combining PAI at several
  coverage levels with R², MAE, and RMSE.
* :func:`temporal_stability` — turn a list of per-period reports into a
  DataFrame so you can see whether the model degrades over time.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _to_1d(a: Any, name: str) -> np.ndarray:
    """Coerce ``a`` to a 1D float numpy array."""
    if isinstance(a, (pd.Series, pd.DataFrame)):
        a = a.to_numpy()
    arr = np.asarray(a, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty.")
    return arr


def _align(preds: Any, actuals: Any) -> tuple[np.ndarray, np.ndarray]:
    """Coerce, validate, and align ``predictions`` and ``actuals``."""
    p = _to_1d(preds, "predictions")
    a = _to_1d(actuals, "actuals")
    if p.shape != a.shape:
        raise ValueError(
            f"predictions and actuals must have the same shape; "
            f"got {p.shape} vs {a.shape}."
        )
    return p, a


def _rank_desc(values: np.ndarray) -> np.ndarray:
    """Return the argsort of ``values`` in descending order.

    Stable sort so ties are broken deterministically by input position.
    """
    return np.argsort(-values, kind="stable")


# ---------------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------------

def predictive_accuracy_index(
    predictions: Any,
    actuals: Any,
    area_pct: float = 0.05,
) -> float:
    """Predictive Accuracy Index (PAI).

    ``PAI = (crime_captured_in_top_area_pct / total_crime) / area_pct``.

    Introduced by Chainey, Tompson & Uhlig (2008); a value of 1 means
    predictions are no better than uniform random targeting, higher is
    better.

    Parameters
    ----------
    predictions:
        Predicted crime counts / scores per cell (higher = hotter).
    actuals:
        Actual crime counts per cell.
    area_pct:
        Fraction of cells to treat as the "hot" area. Default ``0.05``
        (top 5%). Must lie in ``(0, 1]``.

    Returns
    -------
    float
        The PAI. Returns ``nan`` when total crime is zero (no signal to
        capture).
    """
    if not (0.0 < area_pct <= 1.0):
        raise ValueError(
            f"area_pct must be in (0, 1], got {area_pct!r}"
        )
    p, a = _align(predictions, actuals)
    total_crime = a.sum()
    if total_crime <= 0:
        return float("nan")

    n = p.size
    k = max(1, int(np.ceil(n * area_pct)))
    order = _rank_desc(p)
    top_idx = order[:k]
    captured = a[top_idx].sum()

    hit_rate = captured / total_crime
    return float(hit_rate / area_pct)


def hit_rate_curve(
    predictions: Any,
    actuals: Any,
    steps: int = 20,
) -> pd.DataFrame:
    """Compute the hit-rate curve at ``steps`` coverage levels.

    Parameters
    ----------
    predictions, actuals:
        See :func:`predictive_accuracy_index`.
    steps:
        Number of coverage percentages to evaluate, equally spaced in
        ``(0, 1]``. Default ``20`` (i.e. 5%, 10%, ..., 100%).

    Returns
    -------
    pandas.DataFrame
        Columns:

        * ``area_pct`` — fraction of cells targeted.
        * ``crime_pct`` — fraction of total crime captured within those
          cells.
        * ``pai`` — ``crime_pct / area_pct``.
    """
    if steps < 1:
        raise ValueError(f"steps must be >= 1, got {steps}")
    p, a = _align(predictions, actuals)
    n = p.size
    total_crime = a.sum()

    order = _rank_desc(p)
    a_sorted = a[order]
    cum = np.cumsum(a_sorted)

    area_pcts = np.linspace(1.0 / steps, 1.0, steps)
    rows: list[dict[str, float]] = []
    for pct in area_pcts:
        k = max(1, int(np.ceil(n * pct)))
        captured = float(cum[k - 1])
        if total_crime > 0:
            crime_pct = captured / total_crime
            pai = crime_pct / pct
        else:
            crime_pct = float("nan")
            pai = float("nan")
        rows.append(
            {
                "area_pct": float(pct),
                "crime_pct": crime_pct,
                "pai": pai,
            }
        )
    return pd.DataFrame(rows)


def recapture_rate(
    predictions: Any,
    actuals: Any,
    top_n: int | None = None,
    top_pct: float = 0.1,
) -> float:
    """Fraction of actual hotspot cells present in the predicted hotspot cells.

    Both sets are formed by taking the top-K cells (by score for the
    predictions, by observed count for the actuals). This measures
    cell-level agreement and is complementary to PAI (which is
    crime-weighted).

    Parameters
    ----------
    predictions:
        Predicted scores per cell.
    actuals:
        Actual crime counts per cell.
    top_n:
        Explicit number of cells to include in each hotspot set. If
        ``None`` (default), ``top_pct`` is used instead.
    top_pct:
        Fraction of cells to include when ``top_n`` is ``None``. Default
        ``0.1``. Must lie in ``(0, 1]``.

    Returns
    -------
    float
        ``|predicted_top ∩ actual_top| / |actual_top|``. Returns ``nan``
        when the actual-top set is empty (e.g. all zero counts and
        ``top_n=None`` produces a degenerate set of size 0).
    """
    p, a = _align(predictions, actuals)
    n = p.size

    if top_n is not None:
        if top_n <= 0:
            raise ValueError(f"top_n must be positive, got {top_n}")
        k = min(top_n, n)
    else:
        if not (0.0 < top_pct <= 1.0):
            raise ValueError(f"top_pct must be in (0, 1], got {top_pct!r}")
        k = max(1, int(np.ceil(n * top_pct)))

    pred_top = set(_rank_desc(p)[:k].tolist())
    actual_top = set(_rank_desc(a)[:k].tolist())

    if not actual_top:
        return float("nan")
    return len(pred_top & actual_top) / len(actual_top)


def model_report(
    predictions: Any,
    actuals: Any,
    area_pcts: Sequence[float] = (0.01, 0.02, 0.05, 0.10, 0.20),
) -> dict[str, Any]:
    """One-call summary: PAI at several thresholds + regression metrics.

    Parameters
    ----------
    predictions, actuals:
        Predicted and observed counts / scores per cell.
    area_pcts:
        Iterable of coverage fractions at which to report PAI. Default
        ``(0.01, 0.02, 0.05, 0.10, 0.20)``.

    Returns
    -------
    dict
        ``{"pai": {area_pct: value, ...}, "r2": float, "mae": float,
        "rmse": float, "n": int, "total_crime": float}``.
    """
    p, a = _align(predictions, actuals)
    pai_map: dict[float, float] = {}
    for pct in area_pcts:
        pai_map[float(pct)] = predictive_accuracy_index(p, a, area_pct=pct)

    # r2_score requires variance in ``a``; guard for constant targets.
    if np.unique(a).size <= 1:
        r2 = float("nan")
    else:
        r2 = float(r2_score(a, p))
    mae = float(mean_absolute_error(a, p))
    rmse = float(np.sqrt(mean_squared_error(a, p)))

    return {
        "pai": pai_map,
        "r2": r2,
        "mae": mae,
        "rmse": rmse,
        "n": int(p.size),
        "total_crime": float(a.sum()),
    }


def temporal_stability(period_results: Sequence[dict[str, Any]]) -> pd.DataFrame:
    """Assemble per-period :func:`model_report` dicts into a DataFrame.

    Useful for eyeballing whether a model degrades across forecast
    periods. Each row is one period; columns include ``r2``, ``mae``,
    ``rmse``, ``n``, ``total_crime`` and one ``pai@{pct}`` column per
    PAI threshold present in the reports.

    Parameters
    ----------
    period_results:
        Iterable of dicts as returned by :func:`model_report`. Each dict
        may optionally carry a ``"period"`` key which will be preserved
        as a column; otherwise the row index is used.

    Returns
    -------
    pandas.DataFrame
        One row per period. Sorted by ``period`` when that column is
        present and monotonically comparable, otherwise left in input
        order.
    """
    rows: list[dict[str, Any]] = []
    for i, res in enumerate(period_results):
        if not isinstance(res, dict):
            raise TypeError(
                f"period_results[{i}] must be a dict, got {type(res).__name__}"
            )
        row: dict[str, Any] = {
            "period": res.get("period", i),
            "r2": res.get("r2", float("nan")),
            "mae": res.get("mae", float("nan")),
            "rmse": res.get("rmse", float("nan")),
            "n": res.get("n"),
            "total_crime": res.get("total_crime"),
        }
        pai_map = res.get("pai", {}) or {}
        for pct, value in pai_map.items():
            row[f"pai@{float(pct):.2f}"] = value
        rows.append(row)

    df = pd.DataFrame(rows)
    if "period" in df.columns:
        try:
            df = df.sort_values("period").reset_index(drop=True)
        except TypeError:
            # Mixed period types — leave as-is.
            pass
    return df


# ---------------------------------------------------------------------------
# Plot (optional)
# ---------------------------------------------------------------------------

def surveillance_plot(
    predictions: Any,
    actuals: Any,
    steps: int = 20,
    title: str | None = None,
    ax: Any = None,
) -> Any:
    """Plot the hit-rate curve vs. a uniform-random baseline.

    Requires ``matplotlib``. Raises ``ImportError`` if it isn't
    installed — matplotlib is an *optional* dependency of this package
    (``pip install tidycop-hotspots[viz]``).

    Parameters
    ----------
    predictions, actuals:
        Predicted scores and observed counts.
    steps:
        Number of coverage levels to evaluate. Default ``20``.
    title:
        Optional plot title.
    ax:
        Optional pre-existing ``matplotlib.axes.Axes``. A new figure is
        created if ``None``.

    Returns
    -------
    matplotlib.axes.Axes
        The Axes containing the plot.
    """
    try:
        import matplotlib.pyplot as plt  # noqa: F401  (imported for side effects)
    except ImportError as exc:  # pragma: no cover - env-dependent
        raise ImportError(
            "surveillance_plot requires matplotlib. "
            "Install with `pip install tidycop-hotspots[viz]`."
        ) from exc

    curve = hit_rate_curve(predictions, actuals, steps=steps)

    if ax is None:
        import matplotlib.pyplot as plt
        _, ax = plt.subplots(figsize=(6, 5))

    area = curve["area_pct"].to_numpy()
    crime = curve["crime_pct"].to_numpy()

    ax.plot(area, crime, marker="o", label="Model")
    ax.plot([0.0, 1.0], [0.0, 1.0], linestyle="--", color="gray",
            label="Random baseline")
    ax.set_xlabel("Area targeted (fraction of cells)")
    ax.set_ylabel("Crime captured (fraction of total)")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    if title:
        ax.set_title(title)
    return ax


__all__ = [
    "predictive_accuracy_index",
    "hit_rate_curve",
    "recapture_rate",
    "surveillance_plot",
    "model_report",
    "temporal_stability",
]
