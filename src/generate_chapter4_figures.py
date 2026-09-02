#!/usr/bin/env python3
"""Generate the additional map-heavy figures needed for Chapter 4 (Results).

The report-level figures produced by ``generate_report_figures.py`` (Figures
2.1 and 3.1-3.5) are illustrative/methodological. Chapter 4 needs its own,
larger, standalone results maps: multi-year spatial distributions of LST,
NDVI and NDBI; a standalone trend-significance map; a standalone critical
UHI core map; a multi-year TVDI comparison with a severe-stress-area time
series; and the forecast maps presented at full size. This mirrors the
map-per-finding structure used in the sample thesis's Chapter 4.

Figures produced (numbering continues the Chapter 3 figure sequence and is
meant to be renumbered to match wherever these land in the final document):

    Figure 4.1  LST spatial distribution, three-year comparison
    Figure 4.2  NDVI spatial distribution, three-year comparison
    Figure 4.3  NDBI spatial distribution, three-year comparison
    Figure 4.4  LST trend significance (Sen's slope + Mann-Kendall p<0.05), standalone
    Figure 4.5  Critical UHI core (NDBI-LST quadrant overlay), standalone
    Figure 4.6  TVDI spatial distribution, three-year comparison
    Figure 4.7  Severe vegetation stress area, all years (2000-2024) - line chart
    Figure 4.8  SHAP summary - mean absolute feature contribution, XGBoost TVDI model
    Figure 4.9  SHAP dependence - NDBI's marginal contribution to predicted TVDI
    (function figure_4_8_forecast_maps)
                LST forecast maps, MLR and CNN-LSTM-Attention, 2030 and 2040 (full size);
                named figure_4_8_* for historical reasons but corresponds to the final
                document's Figure 4.12 -- do not rename, the docx embedding already
                matches this filename to that caption.

Also written:
    outputs/final/tables/lst_ndvi_ndbi_annual_zonal_stats.csv
        Statewide annual mean/std for LST, NDVI, NDBI (2000-2024) - the
        zonal-statistics table for Chapter 4 (no LGA-level boundary layer
        exists in this project, so statistics are reported state-wide
        rather than per local government area).
    outputs/final/tables/tvdi_severe_stress_area_by_year.csv
        Severe+extreme (TVDI >= --stress-threshold) stress area in km^2
        for every year 2000-2024.

Usage:
    python src/generate_chapter4_figures.py --dpi 600
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import rasterio.mask
import xarray as xr
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from xgboost import Booster, DMatrix

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Chapter 4 results figures and tables")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--figure-output-dir", default="outputs/final/figures", help="Figure output directory")
    parser.add_argument("--table-output-dir", default="outputs/final/tables", help="Table output directory")
    parser.add_argument("--dpi", type=int, default=400, help="PNG export DPI")
    parser.add_argument(
        "--years",
        type=int,
        nargs=3,
        default=[2000, 2012, 2024],
        help="Three years to compare in the multi-year panel figures (default: 2000 2012 2024)",
    )
    parser.add_argument("--stress-threshold", type=float, default=0.6, help="TVDI value at/above which a pixel counts as severe/extreme stress")
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, out_base: Path, dpi: int) -> None:
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def add_north_arrow(ax: plt.Axes, x: float = 0.92, y: float = 0.90) -> None:
    """Standard cartographic north arrow, drawn in axes-fraction coordinates
    so it stays in the same on-screen position regardless of map extent."""
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - 0.09),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=9,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"),
    )


def _nice_scale_length_km(raw_km: float) -> float:
    if raw_km <= 0:
        return 1.0
    magnitude = 10 ** np.floor(np.log10(raw_km))
    for m in (1, 2, 5, 10):
        candidate = m * magnitude
        if candidate >= raw_km:
            return float(candidate)
    return float(10 * magnitude)


def add_scale_bar(ax: plt.Axes) -> None:
    """Graphical scale bar in map units (metres, since all Chapter 4 maps are
    in EPSG:32631). Length is auto-selected to a round number roughly a
    quarter of the visible map width, then drawn on the ground (data)
    coordinates so it is a true, measurable distance on the map."""
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width = xmax - xmin
    height = ymax - ymin

    length_km = _nice_scale_length_km((width * 0.28) / 1000.0)
    length_m = length_km * 1000.0

    x0 = xmin + 0.05 * width
    y0 = ymin + 0.06 * height

    ax.plot([x0, x0 + length_m], [y0, y0], color="black", lw=2.5, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 0.008 * height, y0 + 0.008 * height], color="black", lw=1)
    ax.plot([x0 + length_m, x0 + length_m], [y0 - 0.008 * height, y0 + 0.008 * height], color="black", lw=1)
    ax.text(
        x0 + 0.5 * length_m,
        y0 + 0.015 * height,
        f"{length_km:g} km",
        ha="center",
        va="bottom",
        fontsize=7,
    )


def load_and_clip(path: Path, boundary: gpd.GeoDataFrame) -> tuple[np.ndarray, tuple[float, float, float, float], float]:
    """Open a single-band GeoTIFF, mask it to the Lagos boundary, and return
    (array, extent, pixel_area_km2)."""
    boundary_local = boundary
    with rasterio.open(path) as src:
        if boundary_local.crs != src.crs:
            boundary_local = boundary_local.to_crs(src.crs)
        geoms = [g.__geo_interface__ for g in boundary_local.geometry]
        arr, transform = rasterio.mask.mask(src, geoms, crop=True, nodata=np.nan)
        arr = arr[0].astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        height, width = arr.shape
        left, top = transform * (0, 0)
        right, bottom = transform * (width, height)
        extent = (left, right, bottom, top)
        pixel_area_km2 = abs(transform.a * transform.e) / 1e6
    return arr, extent, pixel_area_km2


def annual_raster_path(project_root: Path, variable: str, year: int) -> Path:
    mapping = {
        "lst": project_root / "data/modis/lst" / f"modis_lst_{year}-01-01.tif",
        "ndvi": project_root / "data/indices/ndvi" / f"modis_ndvi_{year}-01-01.tif",
        "ndbi": project_root / "data/indices/ndbi" / f"modis_ndbi_{year}-01-01.tif",
    }
    return mapping[variable]


def multiyear_panel_figure(
    project_root: Path,
    boundary: gpd.GeoDataFrame,
    variable: str,
    years: list[int],
    cmap: str,
    label: str,
    unit: str,
    out_dir: Path,
    figure_stub: str,
    suptitle: str,
    dpi: int,
    diverging: bool = False,
) -> pd.DataFrame:
    """Build a 1x3 multi-year comparison map for a single variable (LST, NDVI, or NDBI)
    and return a small dataframe of per-year zonal mean/std for the table output."""
    arrays, extents = {}, {}
    for year in years:
        arr, extent, _ = load_and_clip(annual_raster_path(project_root, variable, year), boundary)
        arrays[year] = arr
        extents[year] = extent

    all_vals = np.concatenate([a[np.isfinite(a)] for a in arrays.values()])
    if diverging:
        vabs = float(np.nanpercentile(np.abs(all_vals), 98))
        vmin, vmax = -vabs, vabs
    else:
        vmin = float(np.nanpercentile(all_vals, 2))
        vmax = float(np.nanpercentile(all_vals, 98))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    panel_letters = ["(a)", "(b)", "(c)"]
    boundary_proj = boundary.to_crs(rasterio.open(annual_raster_path(project_root, variable, years[0])).crs)

    im = None
    for ax, year, letter in zip(axes, years, panel_letters):
        im = ax.imshow(arrays[year], extent=extents[year], origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        boundary_proj.boundary.plot(ax=ax, color="black", linewidth=1.0)
        # geopandas' .plot() resets the axis back to equal aspect, so "auto"
        # must be (re-)applied AFTER every geopandas plot call, not before.
        ax.set_aspect("auto")
        ax.set_title(f"{letter} {year}", fontweight="bold")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.grid(alpha=0.12, linestyle=":")
        add_north_arrow(ax)
        add_scale_bar(ax)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label(f"{label} ({unit})" if unit else label)

    fig.suptitle(suptitle, fontsize=14, fontweight="bold")
    save_figure(fig, out_dir / figure_stub, dpi)

    rows = []
    for year in years:
        vals = arrays[year][np.isfinite(arrays[year])]
        rows.append({"year": year, "variable": variable, "mean": float(np.mean(vals)), "std": float(np.std(vals)), "min": float(np.min(vals)), "max": float(np.max(vals))})
    return pd.DataFrame(rows)


def figure_4_4_trend_significance(project_root: Path, out_dir: Path, dpi: int) -> None:
    mk_ds = xr.open_dataset(project_root / "outputs/phase1/mk_sen_maps.nc")
    sen = mk_ds["sen_slope_c_per_year"].values.astype(float)
    pval = mk_ds["mk_pvalue"].values.astype(float)

    if "x" in mk_ds.coords and "y" in mk_ds.coords:
        xvals, yvals = mk_ds["x"].values, mk_ds["y"].values
        extent = (float(np.nanmin(xvals)), float(np.nanmax(xvals)), float(np.nanmin(yvals)), float(np.nanmax(yvals)))
    else:
        raise RuntimeError("mk_sen_maps.nc has no x/y coordinates to derive a map extent from.")

    boundary = gpd.read_file(project_root / "data/vectors/lagos_boundary.geojson").to_crs("EPSG:32631")

    fig, ax = plt.subplots(figsize=(9, 8))
    vabs = float(np.nanpercentile(np.abs(sen[np.isfinite(sen)]), 95))
    im = ax.imshow(sen, extent=extent, origin="upper", cmap="coolwarm", vmin=-vabs, vmax=vabs)
    boundary.boundary.plot(ax=ax, color="black", linewidth=1.3)
    ax.set_aspect("auto")

    sig = np.where(pval < 0.05, 1.0, np.nan)
    cs = ax.contourf(sig, levels=[0.5, 1.5], extent=extent, colors="none", hatches=["////"])
    try:
        hatch_artists = cs.collections
    except AttributeError:
        hatch_artists = [cs]
    for artist in hatch_artists:
        artist.set_edgecolor("#1f2937")
        artist.set_linewidth(0.0)

    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.grid(alpha=0.15, linestyle=":")
    add_north_arrow(ax)
    add_scale_bar(ax)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.1)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Sen's slope (°C/year)")

    ax.set_title(
        "Pixel-Wise Sen's Slope of LST Trend (2000-2024)\nHatched areas: Mann-Kendall p < 0.05",
        fontsize=13,
        fontweight="bold",
    )
    save_figure(fig, out_dir / "figure_4_4_lst_trend_significance", dpi)


def figure_4_5_critical_uhi_core(project_root: Path, out_dir: Path, dpi: int) -> None:
    quad, extent, _ = load_and_clip(
        project_root / "outputs/phase1/ndbi_lst_quadrant_map.tif",
        gpd.read_file(project_root / "data/vectors/lagos_boundary.geojson"),
    )
    stats = pd.read_csv(project_root / "outputs/phase1/ndbi_lst_quadrant_stats.csv")
    boundary = gpd.read_file(project_root / "data/vectors/lagos_boundary.geojson").to_crs("EPSG:32631")

    quad_colors = ["#93c5fd", "#fde68a", "#a7f3d0", "#fca5a5"]
    quad_cmap = matplotlib.colors.ListedColormap(quad_colors)
    quad_norm = matplotlib.colors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], quad_cmap.N)

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.imshow(quad, extent=extent, origin="upper", cmap=quad_cmap, norm=quad_norm)
    boundary.boundary.plot(ax=ax, color="black", linewidth=1.3)
    ax.set_aspect("auto")
    ax.set_xlabel("Easting (m)")
    ax.set_ylabel("Northing (m)")
    ax.grid(alpha=0.15, linestyle=":")
    add_north_arrow(ax)
    add_scale_bar(ax)

    critical_pct = stats.loc[stats["quadrant"] == "High NDBI / High LST", "percentage"].iloc[0]
    labels = [
        "Low NDBI / Low LST",
        "High NDBI / Low LST",
        "Low NDBI / High LST",
        f"High NDBI / High LST\n(critical UHI core, {critical_pct:.1f}%)",
    ]
    handles = [mpatches.Patch(color=quad_colors[i], label=labels[i]) for i in range(4)]
    ax.legend(handles=handles, loc="lower left", frameon=True, fontsize=8)

    ax.set_title("Critical UHI Core: NDBI-LST Quadrant Overlay, Lagos State", fontsize=13, fontweight="bold")
    save_figure(fig, out_dir / "figure_4_5_critical_uhi_core", dpi)


def figure_4_6_tvdi_multiyear(project_root: Path, boundary: gpd.GeoDataFrame, years: list[int], out_dir: Path, dpi: int) -> None:
    arrays, extents = {}, {}
    for year in years:
        arr, extent, _ = load_and_clip(project_root / f"outputs/phase3/tvdi_{year}.tif", boundary)
        arrays[year] = arr
        extents[year] = extent

    boundary_proj = boundary.to_crs(rasterio.open(project_root / f"outputs/phase3/tvdi_{years[0]}.tif").crs)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), constrained_layout=True)
    panel_letters = ["(a)", "(b)", "(c)"]
    im = None
    for ax, year, letter in zip(axes, years, panel_letters):
        im = ax.imshow(arrays[year], extent=extents[year], origin="upper", cmap="YlOrRd", vmin=0, vmax=1)
        boundary_proj.boundary.plot(ax=ax, color="black", linewidth=1.0)
        ax.set_aspect("auto")
        ax.set_title(f"{letter} {year}", fontweight="bold")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.grid(alpha=0.12, linestyle=":")
        add_north_arrow(ax)
        add_scale_bar(ax)

    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("TVDI")
    fig.suptitle("Temperature Vegetation Dryness Index (TVDI), Lagos State", fontsize=14, fontweight="bold")
    save_figure(fig, out_dir / "figure_4_6_tvdi_multiyear", dpi)


def figure_4_7_severe_stress_timeseries(project_root: Path, boundary: gpd.GeoDataFrame, all_years: list[int], threshold: float, out_dir: Path, table_dir: Path, dpi: int) -> None:
    rows = []
    for year in all_years:
        path = project_root / f"outputs/phase3/tvdi_{year}.tif"
        if not path.exists():
            continue
        arr, _, pixel_area_km2 = load_and_clip(path, boundary)
        n_pixels = int(np.nansum(arr >= threshold))
        rows.append({"year": year, "severe_extreme_stress_km2": n_pixels * pixel_area_km2, "severe_extreme_stress_pixels": n_pixels})

    df = pd.DataFrame(rows)
    table_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(table_dir / "tvdi_severe_stress_area_by_year.csv", index=False)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df["year"], df["severe_extreme_stress_km2"], marker="o", color="#b91c1c", linewidth=2)
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Severe + extreme stress area (TVDI ≥ {threshold}), km$^2$")
    ax.set_title("Severe Vegetation Stress Area, Lagos State (2000-2024)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.25, linestyle="--")
    save_figure(fig, out_dir / "figure_4_7_severe_stress_area_timeseries", dpi)


def figure_4_8_shap_summary(project_root: Path, out_dir: Path, dpi: int) -> None:
    """Standalone, full-size SHAP summary bar chart (mean |SHAP| by feature)
    for the XGBoost TVDI model. Reads the already-computed Phase 3 output
    (outputs/phase3/shap_feature_importance.csv) directly, so it does not
    require re-running the Phase 3 pipeline or refetching any raw imagery."""
    shap_df = pd.read_csv(project_root / "outputs/phase3/shap_feature_importance.csv")
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=True)

    label_map = {
        "lst": "LST",
        "ndbi": "NDBI",
        "impervious_fraction": "Impervious fraction",
        "building_density": "Building density",
    }
    labels = [label_map.get(f, f) for f in shap_df["feature"]]

    fig, ax = plt.subplots(figsize=(9, 5.5))
    bars = ax.barh(labels, shap_df["mean_abs_shap"], color="#2563eb")
    for bar, pct in zip(bars, shap_df["contribution_pct"]):
        ax.text(
            bar.get_width() + 0.01 * shap_df["mean_abs_shap"].max(),
            bar.get_y() + bar.get_height() / 2,
            f"{pct:.1f}%",
            va="center",
            ha="left",
            fontsize=9,
        )
    ax.set_xlabel("Mean absolute SHAP value")
    ax.set_ylabel("Feature")
    ax.set_title(
        "SHAP Summary: Mean Absolute Feature Contribution to the\nXGBoost TVDI Model, Lagos State",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.25, linestyle="--")
    save_figure(fig, out_dir / "figure_4_8_shap_summary", dpi)


def figure_4_9_shap_dependence_ndbi(project_root: Path, out_dir: Path, dpi: int) -> None:
    """Standalone, full-size SHAP dependence scatter for NDBI's marginal
    contribution to predicted TVDI. Reloads the already-trained Phase 3
    XGBoost model (outputs/phase3/xgboost_tvdi_model.json) and the
    already-built pixel training table (outputs/phase3/tvdi_training_table.csv)
    and recomputes TreeSHAP contributions on the same held-out test split and
    sample used originally (test = years after --val-end-year, default 2021;
    up to 12,000 rows, random_state=42) -- so no raw MODIS/NDVI/NDBI imagery
    or OSM building-footprint refetch is required."""
    features = ["lst", "ndbi", "impervious_fraction", "building_density"]

    data_df = pd.read_csv(project_root / "outputs/phase3/tvdi_training_table.csv")

    with open(project_root / "outputs/phase3/phase3_metadata.json", encoding="utf-8") as f:
        meta = json.load(f)
    val_end_year = int(meta.get("val_end_year", 2021))

    test = data_df[data_df["year"] > val_end_year]
    X_test = test[features]

    # Load the raw Booster directly (not via the XGBRegressor sklearn wrapper):
    # this only needs the trees themselves for pred_contribs, and sidesteps any
    # sklearn-wrapper metadata incompatibility between the xgboost version that
    # originally trained/saved the model and the one installed now.
    booster = Booster()
    booster.load_model(str(project_root / "outputs/phase3/xgboost_tvdi_model.json"))

    sample_n = min(12000, len(X_test))
    Xs = X_test.sample(n=sample_n, random_state=42) if len(X_test) > sample_n else X_test

    dm = DMatrix(Xs, feature_names=list(Xs.columns))
    contribs = booster.predict(dm, pred_contribs=True)
    shap_values = contribs[:, :-1]  # last column is the bias term

    ndbi_idx = list(Xs.columns).index("ndbi")

    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.scatter(Xs["ndbi"].to_numpy(), shap_values[:, ndbi_idx], s=8, alpha=0.35, color="#2563eb", edgecolors="none")
    ax.axhline(0, color="#6b7280", linewidth=0.8, linestyle="--")
    ax.set_xlabel("NDBI")
    ax.set_ylabel("SHAP value for NDBI")
    ax.set_title(
        "SHAP Dependence: NDBI's Marginal Contribution to\nPredicted TVDI, Lagos State",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(alpha=0.25, linestyle="--")
    save_figure(fig, out_dir / "figure_4_9_shap_dependence_ndbi", dpi)


def figure_4_8_forecast_maps(project_root: Path, out_dir: Path, dpi: int) -> None:
    paths = {
        "MLR 2030": project_root / "outputs/phase2_tf/mlr_lst_forecast_2030.tif",
        "MLR 2040": project_root / "outputs/phase2_tf/mlr_lst_forecast_2040.tif",
        "CNN-LSTM-Attention 2030": project_root / "outputs/phase2_tf/cnn_lstm_attention_lst_forecast_2030.tif",
        "CNN-LSTM-Attention 2040": project_root / "outputs/phase2_tf/cnn_lstm_attention_lst_forecast_2040.tif",
    }
    boundary = gpd.read_file(project_root / "data/vectors/lagos_boundary.geojson")

    arrays, extents = {}, {}
    for key, path in paths.items():
        arr, extent, _ = load_and_clip(path, boundary)
        arrays[key] = arr
        extents[key] = extent

    all_vals = np.concatenate([a[np.isfinite(a)] for a in arrays.values()])
    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))

    boundary_proj = boundary.to_crs(rasterio.open(next(iter(paths.values()))).crs)

    fig, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    im = None
    for ax, key in zip(axes.ravel(), paths.keys()):
        im = ax.imshow(arrays[key], extent=extents[key], origin="upper", cmap="inferno", vmin=vmin, vmax=vmax)
        boundary_proj.boundary.plot(ax=ax, color="white", linewidth=0.8)
        ax.set_aspect("auto")
        ax.set_title(key, fontweight="bold")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.grid(alpha=0.12, linestyle=":")
        add_north_arrow(ax)
        add_scale_bar(ax)

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("Forecasted LST (°C)")
    fig.suptitle("MLR and CNN-LSTM-Attention LST Forecasts for Lagos State (2030 and 2040)", fontsize=14, fontweight="bold")
    save_figure(fig, out_dir / "figure_4_8_lst_forecast_maps", dpi)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    fig_dir = (project_root / args.figure_output_dir).resolve()
    table_dir = (project_root / args.table_output_dir).resolve()
    table_dir.mkdir(parents=True, exist_ok=True)

    apply_style()

    boundary = gpd.read_file(project_root / "data/vectors/lagos_boundary.geojson")
    years = list(args.years)
    all_years = list(range(2000, 2025))

    zonal_rows = []

    zonal_rows.append(
        multiyear_panel_figure(
            project_root, boundary, "lst", years, "inferno", "LST", "°C", fig_dir,
            "figure_4_1_lst_spatial_distribution",
            "Spatial Distribution of Land Surface Temperature, Lagos State",
            args.dpi,
        )
    )
    zonal_rows.append(
        multiyear_panel_figure(
            project_root, boundary, "ndvi", years, "RdYlGn", "NDVI", "", fig_dir,
            "figure_4_2_ndvi_spatial_distribution",
            "Spatial Distribution of NDVI, Lagos State",
            args.dpi,
        )
    )
    zonal_rows.append(
        multiyear_panel_figure(
            project_root, boundary, "ndbi", years, "RdGy_r", "NDBI", "", fig_dir,
            "figure_4_3_ndbi_spatial_distribution",
            "Spatial Distribution of NDBI (Built-Up Intensity), Lagos State",
            args.dpi,
        )
    )
    pd.concat(zonal_rows, ignore_index=True).to_csv(table_dir / "lst_ndvi_ndbi_annual_zonal_stats.csv", index=False)

    figure_4_4_trend_significance(project_root, fig_dir, args.dpi)
    figure_4_5_critical_uhi_core(project_root, fig_dir, args.dpi)
    figure_4_6_tvdi_multiyear(project_root, boundary, years, fig_dir, args.dpi)
    figure_4_7_severe_stress_timeseries(project_root, boundary, all_years, args.stress_threshold, fig_dir, table_dir, args.dpi)
    figure_4_8_shap_summary(project_root, fig_dir, args.dpi)
    figure_4_9_shap_dependence_ndbi(project_root, fig_dir, args.dpi)
    figure_4_8_forecast_maps(project_root, fig_dir, args.dpi)

    print("Chapter 4 figures and tables generated:")
    print(f"  Figures: {fig_dir}")
    print(f"  Tables:  {table_dir}")


if __name__ == "__main__":
    main()
