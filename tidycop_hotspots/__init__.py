"""
tidycop-hotspots — Crime hot spot forecasting with random forests.

Companion package to tidycop. Implements Wheeler & Steenbeek (2021)
machine-learning Risk Terrain Modeling in pure Python, no ArcGIS required.

Modules
-------
grid        Spatial aggregation (square/hex grids, incident counting)
features    Feature engineering (KDE, distance, lagged counts)
model       HotspotForest / HotspotClassifier wrappers around scikit-learn
validate    PAI, hit rate curves, surveillance plots, model reports
"""

__version__ = "0.1.1"

from tidycop_hotspots.grid import (
    make_grid,
    make_hexgrid,
    aggregate_points,
    aggregate_from_df,
)
from tidycop_hotspots.features import (
    kernel_density,
    distance_to_nearest,
    count_within_radius,
    lagged_crime_counts,
    build_feature_matrix,
)
from tidycop_hotspots.model import (
    HotspotForest,
    HotspotClassifier,
    train_test_temporal_split,
)
from tidycop_hotspots.validate import (
    predictive_accuracy_index,
    hit_rate_curve,
    recapture_rate,
    model_report,
)
from tidycop_hotspots.integrations import (
    from_tidycop,
    HotspotBundle,
)

__all__ = [
    # grid
    "make_grid",
    "make_hexgrid",
    "aggregate_points",
    "aggregate_from_df",
    # features
    "kernel_density",
    "distance_to_nearest",
    "count_within_radius",
    "lagged_crime_counts",
    "build_feature_matrix",
    # model
    "HotspotForest",
    "HotspotClassifier",
    "train_test_temporal_split",
    # validate
    "predictive_accuracy_index",
    "hit_rate_curve",
    "recapture_rate",
    "model_report",
    # integrations
    "from_tidycop",
    "HotspotBundle",
]
