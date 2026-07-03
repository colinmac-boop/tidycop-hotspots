"""
CLI entry point for tidycop-hotspots.

Usage:
    tidycop-hotspots grid   --bounds MINX,MINY,MAXX,MAXY --cell-size 250 -o grid.geojson
    tidycop-hotspots train  --grid grid.geojson --features features.csv --target crime_count -o model.joblib
    tidycop-hotspots predict --model model.joblib --grid grid.geojson --features features.csv -o predictions.geojson
    tidycop-hotspots report  --predictions predictions.geojson --actuals actuals.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import geopandas as gpd
import pandas as pd


def cmd_grid(args: argparse.Namespace) -> None:
    """Generate a spatial grid."""
    from tidycop_hotspots.grid import make_grid, make_hexgrid

    bounds = tuple(float(x) for x in args.bounds.split(","))
    if len(bounds) != 4:
        print("Error: --bounds must be MINX,MINY,MAXX,MAXY", file=sys.stderr)
        sys.exit(1)

    fn = make_hexgrid if args.hex else make_grid
    grid = fn(bounds, cell_size_m=args.cell_size, crs=args.crs)

    out = Path(args.output)
    if out.suffix == ".geojson":
        grid.to_file(out, driver="GeoJSON")
    elif out.suffix == ".gpkg":
        grid.to_file(out, driver="GPKG")
    else:
        grid.to_file(out)

    print(f"Wrote {len(grid)} cells to {out}")


def cmd_train(args: argparse.Namespace) -> None:
    """Train a HotspotForest model."""
    from tidycop_hotspots.model import HotspotForest

    grid = gpd.read_file(args.grid)
    features = pd.read_csv(args.features) if args.features else grid

    target = args.target
    if target not in features.columns:
        print(f"Error: target column '{target}' not found", file=sys.stderr)
        sys.exit(1)

    feature_cols = [c for c in features.columns if c != target and c != "cell_id" and c != "geometry"]
    X = features[feature_cols]
    y = features[target]

    model = HotspotForest(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
    )
    model.fit(X, y, feature_names=feature_cols)

    out = Path(args.output)
    model.save(out)
    print(f"Trained model on {len(X)} cells, {len(feature_cols)} features → {out}")

    imp = model.feature_importance()
    print("\nTop 10 features:")
    print(imp.head(10).to_string(index=False))


def cmd_predict(args: argparse.Namespace) -> None:
    """Generate predictions from a trained model."""
    from tidycop_hotspots.model import HotspotForest

    model = HotspotForest.load(args.model)
    grid = gpd.read_file(args.grid)
    features = pd.read_csv(args.features) if args.features else grid

    feature_cols = model.feature_names_
    missing = [c for c in feature_cols if c not in features.columns]
    if missing:
        print(f"Error: missing feature columns: {missing}", file=sys.stderr)
        sys.exit(1)

    X = features[feature_cols]
    grid["predicted_risk"] = model.predict(X)

    out = Path(args.output)
    if out.suffix == ".geojson":
        grid.to_file(out, driver="GeoJSON")
    else:
        grid.to_file(out)

    print(f"Predictions for {len(grid)} cells → {out}")


def cmd_report(args: argparse.Namespace) -> None:
    """Print evaluation report."""
    from tidycop_hotspots.validate import model_report

    pred_gdf = gpd.read_file(args.predictions)
    predictions = pred_gdf["predicted_risk"].values

    if args.actuals.endswith(".csv"):
        actuals_df = pd.read_csv(args.actuals)
        actuals = actuals_df[args.actual_col].values
    else:
        actuals_gdf = gpd.read_file(args.actuals)
        actuals = actuals_gdf[args.actual_col].values

    report = model_report(predictions, actuals)
    print(json.dumps(report, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="tidycop-hotspots",
        description="Crime hot spot forecasting with random forests",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s 0.1.0")
    sub = parser.add_subparsers(dest="command", required=True)

    # grid
    p_grid = sub.add_parser("grid", help="Generate a spatial grid")
    p_grid.add_argument("--bounds", required=True, help="MINX,MINY,MAXX,MAXY")
    p_grid.add_argument("--cell-size", type=float, default=250, help="Cell size in meters")
    p_grid.add_argument("--hex", action="store_true", help="Use hexagonal grid")
    p_grid.add_argument("--crs", default="EPSG:4326", help="Output CRS")
    p_grid.add_argument("-o", "--output", default="grid.geojson")

    # train
    p_train = sub.add_parser("train", help="Train a hotspot model")
    p_train.add_argument("--grid", required=True, help="Grid GeoJSON/GPKG")
    p_train.add_argument("--features", help="Features CSV (or use grid columns)")
    p_train.add_argument("--target", default="crime_count", help="Target column")
    p_train.add_argument("--n-estimators", type=int, default=500)
    p_train.add_argument("--max-depth", type=int, default=None)
    p_train.add_argument("--min-samples-leaf", type=int, default=50)
    p_train.add_argument("-o", "--output", default="model.joblib")

    # predict
    p_predict = sub.add_parser("predict", help="Generate predictions")
    p_predict.add_argument("--model", required=True, help="Trained model .joblib")
    p_predict.add_argument("--grid", required=True, help="Grid GeoJSON/GPKG")
    p_predict.add_argument("--features", help="Features CSV")
    p_predict.add_argument("-o", "--output", default="predictions.geojson")

    # report
    p_report = sub.add_parser("report", help="Evaluation report")
    p_report.add_argument("--predictions", required=True, help="Predictions GeoJSON")
    p_report.add_argument("--actuals", required=True, help="Actuals CSV or GeoJSON")
    p_report.add_argument("--actual-col", default="crime_count")

    args = parser.parse_args()
    {"grid": cmd_grid, "train": cmd_train, "predict": cmd_predict, "report": cmd_report}[args.command](args)


if __name__ == "__main__":
    main()
