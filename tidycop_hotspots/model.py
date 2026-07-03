"""Hot spot prediction models.

Random-forest based regressors and classifiers for spatio-temporal crime
forecasting, plus a temporal train/test split helper. All models are thin
wrappers around scikit-learn estimators with a consistent API (``fit``,
``predict``, ``feature_importance``, ``cross_validate``, ``save``/``load``)
so downstream code can swap regression vs. classification without churn.

The classes here are intentionally sklearn-flavored but *not* full sklearn
estimators (they don't try to pass ``check_estimator``). They store the
feature names that came in at ``fit`` time so ``feature_importance`` returns
a labeled DataFrame instead of a bare array.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _coerce_X(
    X: Any,
    feature_names: Sequence[str] | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Coerce ``X`` to a numpy array and return the inferred feature names.

    Parameters
    ----------
    X:
        Array-like or ``pandas.DataFrame``. 2D expected.
    feature_names:
        Optional explicit feature names. Used when ``X`` is not a DataFrame.

    Returns
    -------
    (values, names):
        The numpy 2D array and the list of column names.
    """
    if isinstance(X, pd.DataFrame):
        names = list(X.columns)
        values = X.to_numpy()
    else:
        values = np.asarray(X)
        if values.ndim == 1:
            values = values.reshape(-1, 1)
        if feature_names is None:
            names = [f"f{i}" for i in range(values.shape[1])]
        else:
            names = list(feature_names)
            if len(names) != values.shape[1]:
                raise ValueError(
                    f"feature_names length {len(names)} != X columns "
                    f"{values.shape[1]}"
                )
    return values, names


def _coerce_y(y: Any) -> np.ndarray:
    """Coerce ``y`` to a 1D numpy array."""
    if isinstance(y, (pd.Series, pd.DataFrame)):
        y = y.to_numpy()
    arr = np.asarray(y)
    if arr.ndim > 1:
        arr = arr.ravel()
    return arr


# ---------------------------------------------------------------------------
# Regressor
# ---------------------------------------------------------------------------

class HotspotForest:
    """Random-forest regressor for predicting crime counts / risk scores.

    Wraps :class:`sklearn.ensemble.RandomForestRegressor` with a small
    convenience layer for feature names, feature importance as a DataFrame,
    k-fold cross validation with regression metrics, and joblib
    persistence.

    Parameters
    ----------
    n_estimators:
        Number of trees in the forest. Default ``500``.
    max_depth:
        Max depth per tree. ``None`` grows unrestricted.
    min_samples_leaf:
        Minimum samples per leaf. Default ``50`` — higher than sklearn's
        default because crime grids are typically noisy and sparse.
    random_state:
        Seed for reproducibility. Default ``42``.
    **kwargs:
        Forwarded to :class:`RandomForestRegressor`.

    Attributes
    ----------
    model:
        The underlying fitted (or unfitted) ``RandomForestRegressor``.
    feature_names_:
        List of feature names captured at ``fit`` time.
    is_fitted:
        Whether ``fit`` has been called successfully.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_leaf: int = 50,
        random_state: int | None = 42,
        **kwargs: Any,
    ) -> None:
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.extra_kwargs = dict(kwargs)

        self.model: RandomForestRegressor = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs,
        )
        self._feature_names: list[str] = []
        self._is_fitted: bool = False

    # ------------------------------------------------------------------ API

    def fit(
        self,
        X: Any,
        y: Any,
        feature_names: Sequence[str] | None = None,
    ) -> "HotspotForest":
        """Fit the forest.

        Parameters
        ----------
        X:
            Array-like or DataFrame of shape ``(n_samples, n_features)``.
        y:
            Array-like of target values (regression targets).
        feature_names:
            Optional list of feature names when ``X`` is not a DataFrame.

        Returns
        -------
        self
        """
        values, names = _coerce_X(X, feature_names)
        y_arr = _coerce_y(y)
        self.model.fit(values, y_arr)
        self._feature_names = names
        self._is_fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict crime counts / risk scores.

        Parameters
        ----------
        X:
            Array-like or DataFrame of shape ``(n_samples, n_features)``.

        Returns
        -------
        numpy.ndarray
            Predicted regression targets, shape ``(n_samples,)``.
        """
        if not self._is_fitted:
            raise RuntimeError("HotspotForest is not fitted; call fit() first.")
        values, _ = _coerce_X(X, self._feature_names)
        return self.model.predict(values)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted descending.

        Returns
        -------
        pandas.DataFrame
            Columns ``["feature", "importance"]``.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "HotspotForest is not fitted; call fit() first."
            )
        df = pd.DataFrame(
            {
                "feature": self._feature_names,
                "importance": self.model.feature_importances_,
            }
        )
        return df.sort_values("importance", ascending=False).reset_index(
            drop=True
        )

    def cross_validate(
        self,
        X: Any,
        y: Any,
        cv: int = 5,
    ) -> dict[str, list[float]]:
        """K-fold cross validation returning per-fold regression metrics.

        This uses a fresh estimator per fold (cloned from the current
        hyperparameters) so calling ``cross_validate`` does *not* fit
        ``self.model``.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Regression targets.
        cv:
            Number of folds. Default ``5``.

        Returns
        -------
        dict
            ``{"r2": [...], "mae": [...], "rmse": [...]}`` with one entry
            per fold.
        """
        values, _ = _coerce_X(X)
        y_arr = _coerce_y(y)
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        r2s: list[float] = []
        maes: list[float] = []
        rmses: list[float] = []

        for train_idx, test_idx in kf.split(values):
            fold_model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                **self.extra_kwargs,
            )
            fold_model.fit(values[train_idx], y_arr[train_idx])
            pred = fold_model.predict(values[test_idx])
            actual = y_arr[test_idx]
            r2s.append(float(r2_score(actual, pred)))
            maes.append(float(mean_absolute_error(actual, pred)))
            rmses.append(float(np.sqrt(mean_squared_error(actual, pred))))

        return {"r2": r2s, "mae": maes, "rmse": rmses}

    # ---------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        """Serialize the estimator to disk with joblib.

        Parameters
        ----------
        path:
            Destination file path.
        """
        payload = {
            "class": "HotspotForest",
            "model": self.model,
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
            "hyperparams": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "random_state": self.random_state,
                "extra_kwargs": self.extra_kwargs,
            },
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "HotspotForest":
        """Load a previously ``save``-d instance.

        Parameters
        ----------
        path:
            Path produced by :meth:`save`.

        Returns
        -------
        HotspotForest
        """
        payload = joblib.load(path)
        if payload.get("class") != "HotspotForest":
            raise ValueError(
                f"File {path} does not contain a HotspotForest payload "
                f"(got {payload.get('class')!r})."
            )
        hp = payload["hyperparams"]
        instance = cls(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            min_samples_leaf=hp["min_samples_leaf"],
            random_state=hp["random_state"],
            **hp.get("extra_kwargs", {}),
        )
        instance.model = payload["model"]
        instance._feature_names = list(payload["feature_names"])
        instance._is_fitted = bool(payload["is_fitted"])
        return instance

    # ------------------------------------------------------------- properties

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    @property
    def n_features_(self) -> int:
        """Number of input features."""
        return len(self._feature_names)

    @property
    def feature_names_(self) -> list[str]:
        """Feature names captured at fit time."""
        return list(self._feature_names)


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

class HotspotClassifier:
    """Random-forest classifier for binary hotspot / not-hotspot labels.

    Wraps :class:`sklearn.ensemble.RandomForestClassifier`. If ``threshold``
    is provided at construction and ``fit`` is called with a continuous
    target, the target is binarized at that percentile: values strictly
    above the percentile become ``1`` (hotspot).

    Parameters
    ----------
    n_estimators, max_depth, min_samples_leaf, random_state, **kwargs:
        See :class:`HotspotForest`.
    threshold:
        Optional float in ``(0, 1)``. Percentile at which to binarize a
        continuous target passed to ``fit``. For example ``0.9`` labels
        the top 10% of cells as hotspots. ``None`` (default) means ``y``
        is assumed to already be binary (or integer class labels).
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int | None = None,
        min_samples_leaf: int = 50,
        random_state: int | None = 42,
        threshold: float | None = None,
        **kwargs: Any,
    ) -> None:
        if threshold is not None and not (0.0 < threshold < 1.0):
            raise ValueError(
                f"threshold must be in (0, 1), got {threshold!r}"
            )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.threshold = threshold
        self.extra_kwargs = dict(kwargs)

        self.model: RandomForestClassifier = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            **kwargs,
        )
        self._feature_names: list[str] = []
        self._is_fitted: bool = False
        self._binarize_cutoff: float | None = None

    # -------------------------------------------------------- binarization

    @staticmethod
    def _looks_binary(y: np.ndarray) -> bool:
        """Return True if ``y`` looks like binary labels already."""
        if y.dtype.kind in ("b",):
            return True
        uniques = np.unique(y)
        if uniques.size <= 2 and set(np.unique(y).tolist()).issubset(
            {0, 1, 0.0, 1.0, True, False}
        ):
            return True
        return False

    def _maybe_binarize(self, y: np.ndarray) -> np.ndarray:
        """Binarize ``y`` at ``self.threshold`` if configured."""
        if self.threshold is None:
            return y.astype(int) if self._looks_binary(y) else y
        if self._looks_binary(y):
            # threshold set but y is already binary — respect y.
            return y.astype(int)
        cutoff = float(np.quantile(y, self.threshold))
        self._binarize_cutoff = cutoff
        return (y > cutoff).astype(int)

    # ------------------------------------------------------------------ API

    def fit(
        self,
        X: Any,
        y: Any,
        feature_names: Sequence[str] | None = None,
    ) -> "HotspotClassifier":
        """Fit the classifier.

        If ``self.threshold`` is set and ``y`` is continuous, ``y`` is
        binarized at that percentile before fitting.

        Parameters
        ----------
        X:
            Array-like or DataFrame of shape ``(n_samples, n_features)``.
        y:
            Array-like target. Binary or continuous.
        feature_names:
            Optional explicit feature names.

        Returns
        -------
        self
        """
        values, names = _coerce_X(X, feature_names)
        y_arr = _coerce_y(y)
        y_bin = self._maybe_binarize(y_arr)
        self.model.fit(values, y_bin)
        self._feature_names = names
        self._is_fitted = True
        return self

    def predict(self, X: Any) -> np.ndarray:
        """Predict binary hotspot labels."""
        if not self._is_fitted:
            raise RuntimeError(
                "HotspotClassifier is not fitted; call fit() first."
            )
        values, _ = _coerce_X(X, self._feature_names)
        return self.model.predict(values)

    def predict_proba(self, X: Any) -> np.ndarray:
        """Predict probability of being a hotspot (class ``1``).

        Parameters
        ----------
        X:
            Feature matrix.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_samples,)`` — the probability of class ``1``.
            If the model was fit on a single-class ``y`` (all zeros or
            all ones), returns the constant probability of that class.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "HotspotClassifier is not fitted; call fit() first."
            )
        values, _ = _coerce_X(X, self._feature_names)
        proba = self.model.predict_proba(values)
        classes = list(self.model.classes_)
        if 1 in classes:
            idx = classes.index(1)
            return proba[:, idx]
        # Only class 0 was seen at fit time.
        return np.zeros(values.shape[0], dtype=float)

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted descending."""
        if not self._is_fitted:
            raise RuntimeError(
                "HotspotClassifier is not fitted; call fit() first."
            )
        df = pd.DataFrame(
            {
                "feature": self._feature_names,
                "importance": self.model.feature_importances_,
            }
        )
        return df.sort_values("importance", ascending=False).reset_index(
            drop=True
        )

    def cross_validate(
        self,
        X: Any,
        y: Any,
        cv: int = 5,
    ) -> dict[str, list[float]]:
        """K-fold cross validation returning per-fold classification metrics.

        Returns per-fold ``accuracy``, ``roc_auc`` (when both classes are
        present in the fold), and ``pr_auc`` (average precision). Falls
        back to NaN for AUC when only one class is present in the test
        fold.

        Parameters
        ----------
        X:
            Feature matrix.
        y:
            Target — binarized on the fly if ``self.threshold`` is set.
        cv:
            Number of folds. Default ``5``.

        Returns
        -------
        dict
            ``{"accuracy": [...], "roc_auc": [...], "pr_auc": [...]}``.
        """
        from sklearn.metrics import (
            accuracy_score,
            average_precision_score,
            roc_auc_score,
        )

        values, _ = _coerce_X(X)
        y_arr = self._maybe_binarize(_coerce_y(y))
        kf = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)

        accs: list[float] = []
        aucs: list[float] = []
        praucs: list[float] = []

        for train_idx, test_idx in kf.split(values):
            fold_model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                **self.extra_kwargs,
            )
            fold_model.fit(values[train_idx], y_arr[train_idx])
            pred = fold_model.predict(values[test_idx])
            actual = y_arr[test_idx]
            accs.append(float(accuracy_score(actual, pred)))

            classes = list(fold_model.classes_)
            if 1 in classes and len(np.unique(actual)) > 1:
                proba = fold_model.predict_proba(values[test_idx])
                pos_idx = classes.index(1)
                score = proba[:, pos_idx]
                aucs.append(float(roc_auc_score(actual, score)))
                praucs.append(float(average_precision_score(actual, score)))
            else:
                aucs.append(float("nan"))
                praucs.append(float("nan"))

        return {"accuracy": accs, "roc_auc": aucs, "pr_auc": praucs}

    # ---------------------------------------------------------- persistence

    def save(self, path: str | Path) -> None:
        """Serialize the estimator to disk with joblib."""
        payload = {
            "class": "HotspotClassifier",
            "model": self.model,
            "feature_names": self._feature_names,
            "is_fitted": self._is_fitted,
            "binarize_cutoff": self._binarize_cutoff,
            "hyperparams": {
                "n_estimators": self.n_estimators,
                "max_depth": self.max_depth,
                "min_samples_leaf": self.min_samples_leaf,
                "random_state": self.random_state,
                "threshold": self.threshold,
                "extra_kwargs": self.extra_kwargs,
            },
        }
        joblib.dump(payload, path)

    @classmethod
    def load(cls, path: str | Path) -> "HotspotClassifier":
        """Load a previously ``save``-d instance."""
        payload = joblib.load(path)
        if payload.get("class") != "HotspotClassifier":
            raise ValueError(
                f"File {path} does not contain a HotspotClassifier payload "
                f"(got {payload.get('class')!r})."
            )
        hp = payload["hyperparams"]
        instance = cls(
            n_estimators=hp["n_estimators"],
            max_depth=hp["max_depth"],
            min_samples_leaf=hp["min_samples_leaf"],
            random_state=hp["random_state"],
            threshold=hp["threshold"],
            **hp.get("extra_kwargs", {}),
        )
        instance.model = payload["model"]
        instance._feature_names = list(payload["feature_names"])
        instance._is_fitted = bool(payload["is_fitted"])
        instance._binarize_cutoff = payload.get("binarize_cutoff")
        return instance

    # ------------------------------------------------------------- properties

    @property
    def is_fitted(self) -> bool:
        """Whether the model has been fitted."""
        return self._is_fitted

    @property
    def n_features_(self) -> int:
        """Number of input features."""
        return len(self._feature_names)

    @property
    def feature_names_(self) -> list[str]:
        """Feature names captured at fit time."""
        return list(self._feature_names)


# ---------------------------------------------------------------------------
# Temporal split
# ---------------------------------------------------------------------------

def train_test_temporal_split(
    grid_data: pd.DataFrame,
    target_col: str,
    feature_cols: Sequence[str],
    train_end_date: Any,
    test_start_date: Any,
    date_col: str = "period",
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Split a spatio-temporal grid dataset by time.

    Random splits leak future information into the past when working with
    space-time data, so *every* spatio-temporal forecasting workflow
    should split on the time axis instead. Training rows have
    ``date_col <= train_end_date``; test rows have
    ``date_col >= test_start_date``. Rows strictly between the two are
    dropped (useful as a gap / holdout buffer).

    Parameters
    ----------
    grid_data:
        DataFrame with one row per (cell, period). Must contain
        ``date_col``, ``target_col``, and all ``feature_cols``.
    target_col:
        Name of the prediction target column.
    feature_cols:
        Names of the feature columns.
    train_end_date:
        Last date (inclusive) to include in the training set. Anything
        comparable to the values in ``date_col`` (string, ``datetime``,
        ``pd.Timestamp``, int).
    test_start_date:
        First date (inclusive) of the test set.
    date_col:
        Name of the date/period column. Default ``"period"``.

    Returns
    -------
    (X_train, y_train, X_test, y_test):
        Feature DataFrames and target Series for the training and test
        windows. Index of the input DataFrame is preserved.

    Raises
    ------
    KeyError
        If any of the required columns is missing.
    ValueError
        If ``test_start_date`` precedes ``train_end_date``, or if the
        resulting training or test window is empty.
    """
    required = {date_col, target_col, *feature_cols}
    missing = required.difference(grid_data.columns)
    if missing:
        raise KeyError(
            f"grid_data is missing required columns: {sorted(missing)!r}"
        )

    # Try to coerce dates to a comparable type when the column looks like
    # dates. Leaves numeric period ids untouched.
    dates = grid_data[date_col]
    is_numeric = pd.api.types.is_numeric_dtype(dates)
    if is_numeric:
        dates_cmp = dates
        train_end_cmp = train_end_date
        test_start_cmp = test_start_date
    else:
        try:
            dates_cmp = pd.to_datetime(dates)
            train_end_cmp = pd.to_datetime(train_end_date)
            test_start_cmp = pd.to_datetime(test_start_date)
        except (ValueError, TypeError):
            dates_cmp = dates
            train_end_cmp = train_end_date
            test_start_cmp = test_start_date

    if test_start_cmp < train_end_cmp:
        raise ValueError(
            f"test_start_date ({test_start_date!r}) must be >= "
            f"train_end_date ({train_end_date!r})"
        )

    train_mask = dates_cmp <= train_end_cmp
    test_mask = dates_cmp >= test_start_cmp

    if not train_mask.any():
        raise ValueError(
            f"No rows have {date_col} <= {train_end_date!r}; "
            "training set is empty."
        )
    if not test_mask.any():
        raise ValueError(
            f"No rows have {date_col} >= {test_start_date!r}; "
            "test set is empty."
        )

    feature_cols_list = list(feature_cols)
    X_train = grid_data.loc[train_mask, feature_cols_list].copy()
    y_train = grid_data.loc[train_mask, target_col].copy()
    X_test = grid_data.loc[test_mask, feature_cols_list].copy()
    y_test = grid_data.loc[test_mask, target_col].copy()

    return X_train, y_train, X_test, y_test


__all__ = [
    "HotspotForest",
    "HotspotClassifier",
    "train_test_temporal_split",
]
