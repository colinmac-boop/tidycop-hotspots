"""
DC Robbery Forecast — Wheeler-style RTM example.

Demonstrates the full tidycop-hotspots pipeline:
1. Generate a spatial grid over Washington DC
2. Simulate incident data (replace with tidycop.get_incidents in production)
3. Engineer features (KDE, distance to POIs)
4. Train a HotspotForest model
5. Evaluate with PAI and hit rate curves

Based on Wheeler & Steenbeek (2021), "Mapping the Risk Terrain for Crime
Using Machine Learning," J. Quantitative Criminology 37(2): 445-480.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

import tidycop_hotspots as th

# --- Configuration ---
DC_BOUNDS = (-77.12, 38.80, -76.91, 38.99)
CELL_SIZE_M = 250
N_TREES = 500
SEED = 42

rng = np.random.default_rng(SEED)


def simulate_incidents(n: int = 2000) -> gpd.GeoDataFrame:
    """
    Simulate crime incidents in DC.

    In production, replace with:
        import tidycop
        df = tidycop.get_incidents("washington_dc", "2024-01-01", "2024-12-31")
    """
    # Create clusters to simulate hotspots
    centers = [(-77.03, 38.90), (-77.00, 38.87), (-76.98, 38.92)]
    points = []
    for cx, cy in centers:
        cluster_n = n // len(centers)
        lons = rng.normal(cx, 0.01, cluster_n)
        lats = rng.normal(cy, 0.01, cluster_n)
        points.extend(Point(lon, lat) for lon, lat in zip(lons, lats))

    # Add uniform background noise
    noise_n = n - len(points)
    lons = rng.uniform(DC_BOUNDS[0], DC_BOUNDS[2], noise_n)
    lats = rng.uniform(DC_BOUNDS[1], DC_BOUNDS[3], noise_n)
    points.extend(Point(lon, lat) for lon, lat in zip(lons, lats))

    return gpd.GeoDataFrame(geometry=points, crs="EPSG:4326")


def simulate_pois(n: int = 50) -> gpd.GeoDataFrame:
    """Simulate points of interest (bars, transit stops, etc.)."""
    lons = rng.uniform(DC_BOUNDS[0], DC_BOUNDS[2], n)
    lats = rng.uniform(DC_BOUNDS[1], DC_BOUNDS[3], n)
    return gpd.GeoDataFrame(
        geometry=[Point(lon, lat) for lon, lat in zip(lons, lats)],
        crs="EPSG:4326",
    )


def main():
    print("=== DC Robbery Forecast Example ===\n")

    # 1. Build grid
    print(f"Building {CELL_SIZE_M}m grid over DC...")
    grid = th.make_grid(DC_BOUNDS, cell_size_m=CELL_SIZE_M)
    print(f"  {len(grid)} grid cells\n")

    # 2. Simulate training data (2024) and test data (2025)
    print("Simulating incident data...")
    train_incidents = simulate_incidents(2000)
    test_incidents = simulate_incidents(1500)
    pois = simulate_pois(50)
    print(f"  Training: {len(train_incidents)} incidents")
    print(f"  Testing:  {len(test_incidents)} incidents")
    print(f"  POIs:     {len(pois)} locations\n")

    # 3. Aggregate to grid
    print("Aggregating incidents to grid cells...")
    grid = th.aggregate_points(grid, train_incidents, count_col="train_count")
    grid = th.aggregate_points(grid, test_incidents, count_col="test_count")
    nonzero = (grid["train_count"] > 0).sum()
    print(f"  {nonzero}/{len(grid)} cells have ≥1 training incident\n")

    # 4. Engineer features
    print("Engineering features...")
    kde = th.kernel_density(grid, train_incidents, bandwidth_m=500)
    dist_poi = th.distance_to_nearest(grid, pois)
    count_poi = th.count_within_radius(grid, pois, radius_m=1000)

    features = th.build_feature_matrix(grid, {
        "kde_crime": kde,
        "dist_poi": dist_poi,
        "count_poi_1km": count_poi,
    })
    print(f"  Feature matrix: {features.shape}\n")

    # 5. Train model
    print(f"Training HotspotForest ({N_TREES} trees)...")
    model = th.HotspotForest(
        n_estimators=N_TREES,
        min_samples_leaf=50,
        random_state=SEED,
    )
    model.fit(features, grid["train_count"])
    print("  Done.\n")

    # Feature importance
    imp = model.feature_importance()
    print("Feature importance:")
    for _, row in imp.iterrows():
        bar = "█" * int(row["importance"] * 50)
        print(f"  {row['feature']:20s} {row['importance']:.4f} {bar}")
    print()

    # 6. Cross-validate
    print("Cross-validating (5-fold)...")
    cv = model.cross_validate(features, grid["train_count"], cv=5)
    print(f"  R²:   {np.mean(cv['r2']):.4f} ± {np.std(cv['r2']):.4f}")
    print(f"  MAE:  {np.mean(cv['mae']):.4f} ± {np.std(cv['mae']):.4f}")
    print(f"  RMSE: {np.mean(cv['rmse']):.4f} ± {np.std(cv['rmse']):.4f}\n")

    # 7. Predict and evaluate against test data
    predictions = model.predict(features)
    report = th.model_report(predictions, grid["test_count"].values)

    pai = report["pai"]
    print("=== Evaluation against held-out test data ===")
    print(f"  R²:       {report['r2']:.4f}")
    print(f"  MAE:      {report['mae']:.4f}")
    print(f"  RMSE:     {report['rmse']:.4f}")
    print(f"  PAI@1%:   {pai[0.01]:.2f}")
    print(f"  PAI@2%:   {pai[0.02]:.2f}")
    print(f"  PAI@5%:   {pai[0.05]:.2f}")
    print(f"  PAI@10%:  {pai[0.10]:.2f}")
    print(f"  PAI@20%:  {pai[0.20]:.2f}")

    # 8. Hit rate curve
    curve = th.hit_rate_curve(predictions, grid["test_count"].values)
    print("\nHit rate curve (area% → crime%):")
    for _, row in curve.iterrows():
        pct = int(row["area_pct"] * 100)
        crime = int(row["crime_pct"] * 100)
        bar = "█" * (crime // 2)
        print(f"  {pct:3d}% area → {crime:3d}% crime {bar}")

    print("\nDone.")


if __name__ == "__main__":
    main()
