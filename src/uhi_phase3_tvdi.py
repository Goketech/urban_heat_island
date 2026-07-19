#!/usr/bin/env python3
"""Phase 3: TVDI modeling, thermal vulnerability, and SHAP attribution.

Workflow:
1) Compute TVDI from annual LST/NDVI using dry/wet edge regressions.
2) Build XGBoost model: TVDI ~ LST + NDBI + impervious_fraction + building_density.
3) Generate SHAP summary/dependence plots and contribution percentages for
   LST and NDBI under extreme vegetation stress.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import osmnx as ox
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from matplotlib import pyplot as plt
from rasterio import features
from rasterio.transform import Affine
from scipy.ndimage import uniform_filter
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import DMatrix, XGBRegressor

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3 TVDI + XGBoost + SHAP")
    parser.add_argument("--lagos-boundary", required=True)
    parser.add_argument("--modis-lst-pattern", required=True)
    parser.add_argument("--ndvi-pattern", required=True)
    parser.add_argument("--ndbi-pattern", required=True)

    parser.add_argument("--impervious-pattern", default=None)
    parser.add_argument("--building-density-raster", default=None)

    parser.add_argument("--osm-buildings-output", default="data/vectors/lagos_osm_buildings.geojson")
    parser.add_argument("--output-dir", required=True)

    parser.add_argument("--tvdi-bins", type=int, default=20)
    parser.add_argument("--train-end-year", type=int, default=2018)
    parser.add_argument("--val-end-year", type=int, default=2021)
    parser.add_argument("--extreme-threshold", type=float, default=0.8)

    parser.add_argument("--xgb-n-estimators", type=int, default=400)
    parser.add_argument("--xgb-learning-rate", type=float, default=0.05)
    parser.add_argument("--xgb-max-depth", type=int, default=6)
    parser.add_argument("--xgb-subsample", type=float, default=0.8)
    parser.add_argument("--xgb-colsample-bytree", type=float, default=0.8)

    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def parse_timestamp(stem: str) -> pd.Timestamp | pd.NaT:
    match = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
    if match:
        return pd.to_datetime(match.group(1), errors="coerce")
    return pd.to_datetime(stem, errors="coerce")


def find_files(pattern: str) -> list[Path]:
    files = sorted(Path().glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    return files


def read_raster_stack(pattern: str) -> xr.DataArray:
    files = find_files(pattern)
    arrs = []
    ts = []
    for f in files:
        da = xr.open_dataarray(f)
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.squeeze("band", drop=True)
        arrs.append(da)
        ts.append(parse_timestamp(f.stem))

    if any(pd.isna(t) for t in ts):
        ts = list(pd.date_range("2000-01-01", periods=len(files), freq="YS"))

    return xr.concat(arrs, dim="time").assign_coords(time=pd.DatetimeIndex(ts))


def load_polygon(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Boundary file is empty")
    return gdf


def clip_to_boundary(da: xr.DataArray, boundary: gpd.GeoDataFrame) -> xr.DataArray:
    target = boundary
    if boundary.crs is None:
        raise ValueError("Boundary CRS is undefined")
    if da.rio.crs is None:
        raise ValueError("Raster CRS is undefined")
    if target.crs != da.rio.crs:
        target = target.to_crs(da.rio.crs)
    return da.rio.clip(target.geometry, target.crs, drop=True)


def annualize(da: xr.DataArray) -> xr.DataArray:
    return da.resample(time="YS").mean(skipna=True)


def build_impervious_proxy(ndbi_y: xr.DataArray) -> xr.DataArray:
    nmin = float(np.nanpercentile(ndbi_y.values, 2))
    nmax = float(np.nanpercentile(ndbi_y.values, 98))
    arr = (ndbi_y - nmin) / (nmax - nmin) if nmax > nmin else xr.zeros_like(ndbi_y)
    return arr.clip(0, 1).rename("impervious_fraction")


def fetch_osm_buildings(boundary: gpd.GeoDataFrame, out_path: Path) -> gpd.GeoDataFrame:
    ox.settings.requests_timeout = 60
    b = boundary.to_crs(4326)
    poly = b.geometry.union_all()
    buildings = ox.features_from_polygon(poly, tags={"building": True})
    if buildings.empty:
        raise ValueError("No OSM buildings returned for Lagos polygon")
    buildings = buildings.reset_index(drop=True)
    buildings = buildings[["geometry"]].copy()
    buildings = buildings[~buildings.geometry.is_empty & buildings.geometry.notna()]
    buildings.to_file(out_path, driver="GeoJSON")
    return buildings


def transform_from_coords(x: np.ndarray, y: np.ndarray) -> Affine:
    if len(x) < 2 or len(y) < 2:
        raise ValueError("Need at least 2 x/y coordinates for raster transform")
    xres = float(abs(x[1] - x[0]))
    yres = float(abs(y[1] - y[0]))
    x0 = float(x.min()) - xres / 2
    y0 = float(y.max()) + yres / 2
    return Affine(xres, 0.0, x0, 0.0, -yres, y0)


def building_density_from_osm(
    boundary: gpd.GeoDataFrame,
    ref_grid: xr.DataArray,
    osm_out_path: Path,
) -> xr.DataArray:
    if osm_out_path.exists():
        bld = gpd.read_file(osm_out_path)
    else:
        bld = fetch_osm_buildings(boundary, osm_out_path)

    if ref_grid.rio.crs is None:
        raise ValueError("Reference grid CRS undefined")

    if bld.crs is None:
        bld = bld.set_crs(4326)
    bld = bld.to_crs(ref_grid.rio.crs)

    x = ref_grid.x.values
    y = ref_grid.y.values
    transform = transform_from_coords(x, y)
    out_shape = (len(y), len(x))

    # Binary built mask from footprints.
    mask = features.rasterize(
        [(geom, 1) for geom in bld.geometry if geom is not None],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="float32",
        all_touched=True,
    )

    # Local mean (3x3) approximates neighborhood building density.
    density = uniform_filter(mask.astype(np.float32), size=3, mode="nearest")
    da = xr.DataArray(density, coords={"y": y, "x": x}, dims=("y", "x"), name="building_density")
    return da.clip(0, 1)


def compute_tvdi_for_year(lst: np.ndarray, ndvi: np.ndarray, bins: int):
    valid = np.isfinite(lst) & np.isfinite(ndvi)
    if valid.sum() < bins * 20:
        return np.full_like(lst, np.nan), np.nan, np.nan, np.nan, np.nan

    lst_v = lst[valid]
    ndvi_v = ndvi[valid]

    qmin, qmax = np.nanpercentile(ndvi_v, [2, 98])
    edges = np.linspace(qmin, qmax, bins + 1)

    dry_x, dry_y, wet_x, wet_y = [], [], [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i + 1]
        m = (ndvi_v >= lo) & (ndvi_v < hi)
        if m.sum() < 10:
            continue
        nd_bin = ndvi_v[m]
        ls_bin = lst_v[m]
        dry_x.append(float(np.nanmean(nd_bin)))
        wet_x.append(float(np.nanmean(nd_bin)))
        dry_y.append(float(np.nanpercentile(ls_bin, 95)))
        wet_y.append(float(np.nanpercentile(ls_bin, 5)))

    if len(dry_x) < 3 or len(wet_x) < 3:
        return np.full_like(lst, np.nan), np.nan, np.nan, np.nan, np.nan

    b_dry, a_dry = np.polyfit(dry_x, dry_y, 1)
    b_wet, a_wet = np.polyfit(wet_x, wet_y, 1)

    lst_max = a_dry + b_dry * ndvi
    lst_min = a_wet + b_wet * ndvi
    denom = lst_max - lst_min
    tvdi = (lst - lst_min) / denom
    tvdi = np.where(np.isfinite(tvdi), np.clip(tvdi, 0, 1), np.nan)
    return tvdi, a_dry, b_dry, a_wet, b_wet


def compute_tvdi_stack(lst_y: xr.DataArray, ndvi_y: xr.DataArray, bins: int):
    tvdi_maps = []
    edge_rows = []

    for year in lst_y.year.values:
        l = lst_y.sel(year=year).values
        n = ndvi_y.sel(year=year).values
        tvdi, a_dry, b_dry, a_wet, b_wet = compute_tvdi_for_year(l, n, bins=bins)
        tvdi_maps.append(tvdi)
        edge_rows.append(
            {
                "year": int(year),
                "a_dry": a_dry,
                "b_dry": b_dry,
                "a_wet": a_wet,
                "b_wet": b_wet,
            }
        )

    tvdi_arr = np.stack(tvdi_maps, axis=0)
    tvdi_da = xr.DataArray(
        tvdi_arr,
        coords={"year": lst_y.year.values, "y": lst_y.y.values, "x": lst_y.x.values},
        dims=("year", "y", "x"),
        name="TVDI",
    )
    return tvdi_da, pd.DataFrame(edge_rows)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def build_training_table(
    years: np.ndarray,
    lst_y: xr.DataArray,
    ndbi_y: xr.DataArray,
    tvdi_y: xr.DataArray,
    imp_y: xr.DataArray,
    bld_y: xr.DataArray,
) -> pd.DataFrame:
    frames = []
    for year in years:
        l = lst_y.sel(year=year)
        n = ndbi_y.sel(year=year)
        t = tvdi_y.sel(year=year)
        i = imp_y.sel(year=year)
        b = bld_y.sel(year=year)
        l, n, t, i, b = xr.align(l, n, t, i, b, join="inner")

        arr = {
            "lst": l.values.ravel(),
            "ndbi": n.values.ravel(),
            "tvdi": t.values.ravel(),
            "impervious_fraction": i.values.ravel(),
            "building_density": b.values.ravel(),
        }
        df = pd.DataFrame(arr)
        df["year"] = int(year)
        df = df.replace([np.inf, -np.inf], np.nan).dropna()
        if not df.empty:
            frames.append(df)

    if not frames:
        raise ValueError("No valid rows available for TVDI model training")
    return pd.concat(frames, ignore_index=True)


def save_shap_outputs(model, X_test: pd.DataFrame, y_test: pd.Series, out_dir: Path, extreme_threshold: float):
    sample_n = min(12000, len(X_test))
    Xs = X_test.sample(n=sample_n, random_state=42) if len(X_test) > sample_n else X_test
    ys = y_test.loc[Xs.index]

    booster = model.get_booster()
    dm = DMatrix(Xs, feature_names=list(Xs.columns))
    contribs = booster.predict(dm, pred_contribs=True)
    # Last column is bias term; exclude from feature attributions.
    shap_values = contribs[:, :-1]

    abs_mean = np.abs(shap_values).mean(axis=0)
    order = np.argsort(abs_mean)[::-1]
    ordered_features = Xs.columns.to_numpy()[order]
    ordered_abs = abs_mean[order]

    # Summary plot: mean absolute SHAP contribution bars.
    plt.figure(figsize=(9, 6))
    plt.barh(ordered_features[::-1], ordered_abs[::-1])
    plt.xlabel("Mean |SHAP value|")
    plt.ylabel("Feature")
    plt.title("SHAP Summary (TreeSHAP via XGBoost)")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_summary.png", dpi=220)
    plt.close()

    # Dependence plot for NDBI.
    ndbi_idx = list(Xs.columns).index("ndbi")
    plt.figure(figsize=(9, 6))
    plt.scatter(Xs["ndbi"].to_numpy(), shap_values[:, ndbi_idx], s=6, alpha=0.35)
    plt.xlabel("NDBI")
    plt.ylabel("SHAP value for NDBI")
    plt.title("SHAP Dependence: NDBI -> TVDI")
    plt.tight_layout()
    plt.savefig(out_dir / "shap_dependence_ndbi.png", dpi=220)
    plt.close()

    feature_importance = pd.DataFrame(
        {
            "feature": Xs.columns,
            "mean_abs_shap": abs_mean,
            "contribution_pct": 100.0 * abs_mean / abs_mean.sum(),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    feature_importance.to_csv(out_dir / "shap_feature_importance.csv", index=False)

    extreme_mask = ys.to_numpy() >= extreme_threshold
    if extreme_mask.sum() > 0:
        abs_ext = np.abs(shap_values[extreme_mask]).mean(axis=0)
        ext_df = pd.DataFrame(
            {
                "feature": Xs.columns,
                "mean_abs_shap_extreme": abs_ext,
                "contribution_pct_extreme": 100.0 * abs_ext / abs_ext.sum(),
            }
        ).sort_values("mean_abs_shap_extreme", ascending=False)
    else:
        ext_df = pd.DataFrame(
            {
                "feature": Xs.columns,
                "mean_abs_shap_extreme": np.nan,
                "contribution_pct_extreme": np.nan,
            }
        )

    ext_df.to_csv(out_dir / "shap_extreme_feature_contributions.csv", index=False)

    focus = ext_df[ext_df["feature"].isin(["lst", "ndbi"])]
    focus.to_csv(out_dir / "lst_ndbi_extreme_contribution_pct.csv", index=False)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    boundary = load_polygon(args.lagos_boundary)

    lst = clip_to_boundary(read_raster_stack(args.modis_lst_pattern), boundary)
    ndvi = clip_to_boundary(read_raster_stack(args.ndvi_pattern), boundary)
    ndbi = clip_to_boundary(read_raster_stack(args.ndbi_pattern), boundary)

    lst_y = annualize(lst)
    ndvi_y = annualize(ndvi)
    ndbi_y = annualize(ndbi)

    # Harmonize NDBI/NDVI to LST grid.
    ndvi_y = ndvi_y.rio.reproject_match(lst_y)
    ndbi_y = ndbi_y.rio.reproject_match(lst_y)

    years = np.intersect1d(np.intersect1d(lst_y.time.dt.year.values, ndvi_y.time.dt.year.values), ndbi_y.time.dt.year.values)

    lst_y = lst_y.assign_coords(year=lst_y.time.dt.year).swap_dims({"time": "year"}).drop_vars("time").sel(year=years)
    ndvi_y = ndvi_y.assign_coords(year=ndvi_y.time.dt.year).swap_dims({"time": "year"}).drop_vars("time").sel(year=years)
    ndbi_y = ndbi_y.assign_coords(year=ndbi_y.time.dt.year).swap_dims({"time": "year"}).drop_vars("time").sel(year=years)

    tvdi_y, edge_df = compute_tvdi_stack(lst_y, ndvi_y, bins=args.tvdi_bins)
    edge_df.to_csv(out_dir / "tvdi_dry_wet_edges_by_year.csv", index=False)

    tvdi_y.to_netcdf(out_dir / "tvdi_stack.nc")
    for year in tvdi_y.year.values:
        da = tvdi_y.sel(year=year)
        da.rio.write_crs(lst.rio.crs, inplace=True)
        da.rio.to_raster(out_dir / f"tvdi_{int(year)}.tif")

    # Impervious fraction
    impervious_source = "proxy_from_ndbi"
    if args.impervious_pattern:
        imp = clip_to_boundary(read_raster_stack(args.impervious_pattern), boundary)
        imp_y = annualize(imp)
        imp_y = imp_y.rio.reproject_match(lst_y)
        imp_y = imp_y.assign_coords(year=imp_y.time.dt.year).swap_dims({"time": "year"}).drop_vars("time")
        if len(imp_y.year.values) == 1:
            imp_y = xr.concat([imp_y.isel(year=0)] * len(years), dim="year").assign_coords(year=years)
        else:
            imp_y = imp_y.reindex(year=years).interpolate_na(dim="year", method="linear", fill_value="extrapolate")
        impervious_source = "provided_raster"
    else:
        imp_y = build_impervious_proxy(ndbi_y).rename("impervious_fraction")

    # Building density
    building_source = "osm_footprints_density"
    if args.building_density_raster:
        bd = clip_to_boundary(read_raster_stack(args.building_density_raster), boundary)
        bd_y = annualize(bd)
        bd_y = bd_y.rio.reproject_match(lst_y)
        bd_y = bd_y.assign_coords(year=bd_y.time.dt.year).swap_dims({"time": "year"}).drop_vars("time")
        if len(bd_y.year.values) == 1:
            bd_y = xr.concat([bd_y.isel(year=0)] * len(years), dim="year").assign_coords(year=years)
        else:
            bd_y = bd_y.reindex(year=years).interpolate_na(dim="year", method="linear", fill_value="extrapolate")
        building_source = "provided_raster"
    else:
        try:
            bd_static = building_density_from_osm(boundary, lst_y.isel(year=0), Path(args.osm_buildings_output))
        except Exception:
            # Fallback: use smoothed impervious proxy where OSM retrieval is unavailable.
            proxy = np.nan_to_num(imp_y.isel(year=0).values.astype(np.float32), nan=0.0)
            smoothed = uniform_filter(proxy, size=3, mode="nearest")
            bd_static = xr.DataArray(
                smoothed,
                coords={"y": lst_y.y.values, "x": lst_y.x.values},
                dims=("y", "x"),
                name="building_density",
            ).clip(0, 1)
            building_source = "ndbi_proxy_fallback"
        bd_y = xr.concat([bd_static] * len(years), dim="year").assign_coords(year=years)

    data_df = build_training_table(years, lst_y, ndbi_y, tvdi_y, imp_y, bd_y)
    data_df.to_csv(out_dir / "tvdi_training_table.csv", index=False)

    train = data_df[data_df["year"] <= args.train_end_year]
    val = data_df[(data_df["year"] > args.train_end_year) & (data_df["year"] <= args.val_end_year)]
    test = data_df[data_df["year"] > args.val_end_year]

    features = ["lst", "ndbi", "impervious_fraction", "building_density"]
    X_train, y_train = train[features], train["tvdi"]
    X_val, y_val = val[features], val["tvdi"]
    X_test, y_test = test[features], test["tvdi"]

    model = XGBRegressor(
        n_estimators=args.xgb_n_estimators,
        learning_rate=args.xgb_learning_rate,
        max_depth=args.xgb_max_depth,
        subsample=args.xgb_subsample,
        colsample_bytree=args.xgb_colsample_bytree,
        objective="reg:squarederror",
        random_state=args.random_state,
        n_jobs=4,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    rows = []
    for split, X, y in [("train", X_train, y_train), ("val", X_val, y_val), ("test", X_test, y_test)]:
        pred = model.predict(X)
        rows.append({"split": split, **evaluate(y.to_numpy(), pred)})
    pd.DataFrame(rows).to_csv(out_dir / "xgboost_tvdi_metrics.csv", index=False)

    model.save_model(str(out_dir / "xgboost_tvdi_model.json"))

    save_shap_outputs(model, X_test, y_test, out_dir=out_dir, extreme_threshold=args.extreme_threshold)

    metadata = {
        "impervious_source": impervious_source,
        "building_density_source": building_source,
        "extreme_threshold": args.extreme_threshold,
        "train_end_year": args.train_end_year,
        "val_end_year": args.val_end_year,
    }
    (out_dir / "phase3_metadata.json").write_text(pd.Series(metadata).to_json(indent=2), encoding="utf-8")

    print("Phase 3 TVDI analysis complete.")
    print(f"Outputs written to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
