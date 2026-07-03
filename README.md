# tidycop-hotspots

**Crime hot spot forecasting with random forests** — companion package to
[tidycop](https://github.com/colinmac-boop/tidycop).

Implements the machine-learning Risk Terrain Modeling methodology from
[Wheeler & Steenbeek (2021)](https://link.springer.com/article/10.1007/s10940-020-09457-7)
in pure Python. No ArcGIS Pro required.

## What It Does

1. **Spatial aggregation** — grid a city into square or hexagonal cells,
   count incidents per cell
2. **Feature engineering** — kernel density estimates, distance to points
   of interest, lagged crime counts, temporal features
3. **Hot spot prediction** — train random forest models to forecast where
   crime concentrates
4. **Validation** — Predictive Accuracy Index (PAI), hit rate curves,
   surveillance plots

## Install

```bash
pip install tidycop-hotspots

# With tidycop integration + visualization
pip install "tidycop-hotspots[all]"
```

## Quick Start

```python
import tidycop
import tidycop_hotspots as th

# 1. Pull crime data
incidents = tidycop.get_incidents("chicago", "2025-01-01", "2025-12-31")

# 2. Build a grid over Chicago
bounds = (-87.94, 41.64, -87.52, 42.02)  # Chicago bbox
grid = th.make_grid(bounds, cell_size_m=250)

# 3. Count crimes per cell
grid = th.aggregate_from_df(grid, incidents)

# 4. Engineer features
kde = th.kernel_density(grid, incidents_gdf)
dist_transit = th.distance_to_nearest(grid, transit_stops_gdf)
features = th.build_feature_matrix(grid, {
    "kde_crime": kde,
    "dist_transit": dist_transit,
})

# 5. Train model
model = th.HotspotForest(n_estimators=500, min_samples_leaf=50)
model.fit(features, grid["crime_count"])

# 6. Evaluate
future_crimes = tidycop.get_incidents("chicago", "2026-01-01", "2026-06-30")
future_grid = th.aggregate_from_df(grid, future_crimes, count_col="future_count")
report = th.model_report(model.predict(features), future_grid["future_count"])
print(f"PAI@5%: {report['pai_0.05']:.2f}")
```

## CLI

```bash
# Generate grid
tidycop-hotspots grid --bounds -87.94,41.64,-87.52,42.02 --cell-size 250 -o chicago_grid.geojson

# Train
tidycop-hotspots train --grid chicago_grid.geojson --target crime_count -o model.joblib

# Predict
tidycop-hotspots predict --model model.joblib --grid chicago_grid.geojson -o predictions.geojson

# Evaluate
tidycop-hotspots report --predictions predictions.geojson --actuals future_counts.csv
```

## Methodology

Based on Wheeler & Steenbeek (2021), "Mapping the Risk Terrain for Crime
Using Machine Learning," *Journal of Quantitative Criminology* 37(2): 445-480.

Key principles:
- **Spatial units** include zero-crime cells (critical for valid predictions)
- **Temporal validation** — always split by time, never random
- **Feature engineering** over raw coordinates — KDE, distance, density
- **Random forests** handle nonlinear interactions between risk factors
- **PAI** (Predictive Accuracy Index) as the primary evaluation metric

## Modules

| Module | Purpose |
|--------|---------|
| `grid` | Square/hex grid generation, incident aggregation |
| `features` | KDE, distance-to-POI, lagged counts, temporal features |
| `model` | `HotspotForest` and `HotspotClassifier` wrappers |
| `validate` | PAI, hit rate curves, surveillance plots, model reports |

## References

- Wheeler, A. P., & Steenbeek, W. (2021). Mapping the risk terrain for
  crime using machine learning. *Journal of Quantitative Criminology*,
  37(2), 445-480.
- Wheeler, A. P. (2024). *Data Science for Crime Analysis with Python*.
- [Using Random Forests in ArcPro to forecast hot spots](https://andrewpwheeler.com/2021/03/26/using-random-forests-in-arcpro-to-forecast-hot-spots/)

## License

MIT
