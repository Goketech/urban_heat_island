#!/usr/bin/env python3
"""Generate publication-quality figures for the Lagos UHI report.

This script exports print-ready PNG and SVG figures for:
- Figure 2.1: Driver-response conceptual framework
- Figure 3.1: Lagos study area map with Nigeria context
- Figure 3.2: Methodological framework
- Figure 3.3: Phase 1 trend and quadrant maps
- Figure 3.4: Phase 2 TVDI + SHAP summary panel
- Figure 3.5: Phase 3 LST forecast maps
"""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import matplotlib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
import xarray as xr
from matplotlib.gridspec import GridSpec
from matplotlib.patches import FancyArrowPatch, Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable
from shapely.geometry import Polygon

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate publication-quality report figures")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    parser.add_argument("--output-dir", default="outputs/final/figures", help="Figure output directory")
    parser.add_argument("--dpi", type=int, default=400, help="PNG export DPI")
    return parser.parse_args()


def apply_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
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


def open_raster(path: Path) -> tuple[np.ndarray, tuple[float, float, float, float], str | None]:
    with rasterio.open(path) as src:
        arr = src.read(1).astype(float)
        nodata = src.nodata
        if nodata is not None:
            arr = np.where(arr == nodata, np.nan, arr)
        bounds = src.bounds
        extent = (bounds.left, bounds.right, bounds.bottom, bounds.top)
        crs = src.crs.to_string() if src.crs is not None else None
    return arr, extent, crs


def add_north_arrow(ax: plt.Axes, x: float = 0.95, y: float = 0.92) -> None:
    ax.annotate(
        "N",
        xy=(x, y),
        xytext=(x, y - 0.08),
        xycoords="axes fraction",
        ha="center",
        va="center",
        fontsize=10,
        fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", lw=1.4, color="black"),
    )


def add_scale_bar(ax: plt.Axes, length_km: float = 20.0) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width = xmax - xmin
    height = ymax - ymin

    x0 = xmin + 0.06 * width
    y0 = ymin + 0.05 * height
    length_m = length_km * 1000.0

    ax.plot([x0, x0 + length_m], [y0, y0], color="black", lw=3, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 0.01 * height, y0 + 0.01 * height], color="black", lw=1)
    ax.plot([x0 + length_m, x0 + length_m], [y0 - 0.01 * height, y0 + 0.01 * height], color="black", lw=1)
    ax.text(x0 + 0.5 * length_m, y0 + 0.018 * height, f"{int(length_km)} km", ha="center", va="bottom", fontsize=9)


def draw_box(ax: plt.Axes, xy: tuple[float, float], w: float, h: float, color: str, text: str) -> None:
    box = Rectangle(xy, w, h, linewidth=1.5, edgecolor="#222222", facecolor=color)
    ax.add_patch(box)
    ax.text(xy[0] + w / 2, xy[1] + h / 2, text, ha="center", va="center", fontsize=10, wrap=True)


def draw_arrow(ax: plt.Axes, p1: tuple[float, float], p2: tuple[float, float], text: str | None = None) -> None:
    arr = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=15, linewidth=1.6, color="#333333")
    ax.add_patch(arr)
    if text:
        mx = (p1[0] + p2[0]) / 2
        my = (p1[1] + p2[1]) / 2
        ax.text(mx, my + 0.015, text, ha="center", va="center", fontsize=9, color="#333333")


def figure_2_1_conceptual(out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(12, 7))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(
        ax,
        (0.05, 0.62),
        0.22,
        0.22,
        "#dbeafe",
        "Urban Expansion\nand Population\nPressure",
    )
    draw_box(
        ax,
        (0.35, 0.62),
        0.22,
        0.22,
        "#fee2e2",
        "Built-up Growth\n(Imperviousness,\nNDBI Increase)",
    )
    draw_box(
        ax,
        (0.65, 0.62),
        0.22,
        0.22,
        "#dcfce7",
        "Vegetation Loss\n(NDVI Decline)",
    )

    draw_box(
        ax,
        (0.25, 0.26),
        0.22,
        0.22,
        "#fef3c7",
        "Rising Land Surface\nTemperature\n(UHI Intensification)",
    )
    draw_box(
        ax,
        (0.55, 0.26),
        0.22,
        0.22,
        "#fde68a",
        "Vegetation Thermal\nStress\n(TVDI Increase)",
    )

    draw_arrow(ax, (0.27, 0.73), (0.35, 0.73))
    draw_arrow(ax, (0.57, 0.73), (0.65, 0.73))
    draw_arrow(ax, (0.46, 0.62), (0.36, 0.48), "Heat amplification")
    draw_arrow(ax, (0.76, 0.62), (0.66, 0.48), "Cooling loss")
    draw_arrow(ax, (0.47, 0.37), (0.55, 0.37), "Stress propagation")

    ax.text(
        0.5,
        0.93,
        "Driver-Response Framework: Urbanization, Thermal Intensification, and Vegetation Stress",
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
    )
    ax.text(
        0.5,
        0.08,
        "Causal direction shows how urban growth influences land-surface heating and ecosystem thermal vulnerability in Lagos.",
        ha="center",
        va="center",
        fontsize=10,
        color="#444444",
    )

    save_figure(fig, out_dir / "figure_2_1_conceptual_framework", dpi)


# Simplified but real Nigeria national boundary (WGS84), used for the study-area
# inset map so it shows the country's actual coastline/border shape rather than
# a placeholder rectangle. Source: public-domain low-resolution country outline
# (comparable to Natural Earth 110m), embedded here so figure generation does
# not depend on an external download at runtime.
NIGERIA_BOUNDARY_WGS84 = [
    (8.500288, 4.771983), (7.462108, 4.412108), (7.082596, 4.464689),
    (6.698072, 4.240594), (5.898173, 4.262453), (5.362805, 4.887971),
    (5.033574, 5.611802), (4.325607, 6.270651), (3.57418, 6.2583),
    (2.691702, 6.258817), (2.749063, 7.870734), (2.723793, 8.506845),
    (2.912308, 9.137608), (3.220352, 9.444153), (3.705438, 10.06321),
    (3.60007, 10.332186), (3.797112, 10.734746), (3.572216, 11.327939),
    (3.61118, 11.660167), (3.680634, 12.552903), (3.967283, 12.956109),
    (4.107946, 13.531216), (4.368344, 13.747482), (5.443058, 13.865924),
    (6.445426, 13.492768), (6.820442, 13.115091), (7.330747, 13.098038),
    (7.804671, 13.343527), (9.014933, 12.826659), (9.524928, 12.851102),
    (10.114814, 13.277252), (10.701032, 13.246918), (10.989593, 13.387323),
    (11.527803, 13.32898), (12.302071, 13.037189), (13.083987, 13.596147),
    (13.318702, 13.556356), (13.995353, 12.461565), (14.181336, 12.483657),
    (14.577178, 12.085361), (14.468192, 11.904752), (14.415379, 11.572369),
    (13.57295, 10.798566), (13.308676, 10.160362), (13.1676, 9.640626),
    (12.955468, 9.417772), (12.753672, 8.717763), (12.218872, 8.305824),
    (12.063946, 7.799808), (11.839309, 7.397042), (11.745774, 6.981383),
    (11.058788, 6.644427), (10.497375, 7.055358), (10.118277, 7.03877),
    (9.522706, 6.453482), (9.233163, 6.444491), (8.757533, 5.479666),
    (8.500288, 4.771983),
]


def figure_3_1_study_area(project_root: Path, out_dir: Path, dpi: int) -> None:
    boundary_path = project_root / "data/vectors/lagos_boundary.geojson"
    boundary = gpd.read_file(boundary_path)
    if boundary.crs is None:
        boundary = boundary.set_crs("EPSG:4326")

    lagos_4326 = boundary.to_crs("EPSG:4326")
    lagos_utm = boundary.to_crs("EPSG:32631")

    # Lagos State's real footprint is a very elongated coastal strip (about
    # 4.9:1, width:height, in UTM metres). Rather than force it into a
    # squarer panel (which either leaves large blank margins at equal aspect,
    # or visibly distorts the recognisable shape at "auto" aspect), the main
    # panel is sized so its own box already matches the true data aspect
    # ratio - the shape is shown undistorted, at true scale, filling its
    # frame with no wasted space. This intentionally makes the figure a wide
    # banner, matching the same elongated shape seen in the Chapter 4 result
    # maps (LST/NDVI/NDBI etc.), which are of the same study area.
    minx, miny, maxx, maxy = lagos_utm.total_bounds
    w, h = maxx - minx, maxy - miny
    xpad, ypad = w * 0.12, h * 0.12
    data_aspect = (w + 2 * xpad) / (h + 2 * ypad)

    fig_h = 6.0
    fig_w = data_aspect * fig_h * 1.03
    fig, ax_main = plt.subplots(figsize=(fig_w, fig_h))

    lagos_utm.plot(ax=ax_main, color="#fca5a5", edgecolor="#7f1d1d", linewidth=1.8)
    ax_main.set_xlim(minx - xpad, maxx + xpad)
    ax_main.set_ylim(miny - ypad, maxy + ypad)
    ax_main.set_xlabel("Easting (m)")
    ax_main.set_ylabel("Northing (m)")
    ax_main.set_title("Lagos State Boundary (Study Area, EPSG:32631)", fontweight="bold")
    ax_main.grid(alpha=0.25, linestyle="--")

    add_north_arrow(ax_main)
    add_scale_bar(ax_main, length_km=20)

    # Nigeria locator inset: a small, correctly-proportioned corner inset
    # (standard cartographic convention) rather than a full-height side
    # panel. Sizing it as a fraction of the main axes, corrected for the
    # main axes' own (very elongated) on-screen aspect ratio, keeps
    # Nigeria's true shape (about 1.24:1) undistorted rather than stretched
    # thin and tall to fill a mismatched box.
    nigeria_poly = Polygon(NIGERIA_BOUNDARY_WGS84)
    nigeria_gdf = gpd.GeoDataFrame({"name": ["Nigeria"]}, geometry=[nigeria_poly], crs="EPSG:4326")
    nminx, nminy, nmaxx, nmaxy = nigeria_poly.bounds
    nigeria_aspect = (nmaxx - nminx) / (nmaxy - nminy)

    width_frac = 0.16
    height_frac = width_frac * data_aspect / nigeria_aspect
    x0, y0 = 0.015, 0.90 - height_frac
    ax_inset = ax_main.inset_axes([x0, y0, width_frac, height_frac])

    # Pale ocean/neighbouring-territory backdrop so Nigeria's coastline and
    # border reads clearly as a real country shape, not a schematic box.
    ax_inset.set_facecolor("#eaf4fb")
    nigeria_gdf.plot(ax=ax_inset, color="#e2e8f0", edgecolor="#334155", linewidth=1.0, zorder=1)
    lagos_4326.plot(ax=ax_inset, color="#dc2626", edgecolor="#7f1d1d", linewidth=0.8, zorder=3)

    npad_x = (nmaxx - nminx) * 0.08
    npad_y = (nmaxy - nminy) * 0.08
    ax_inset.set_xlim(nminx - npad_x, nmaxx + npad_x)
    ax_inset.set_ylim(nminy - npad_y, nmaxy + npad_y)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    ax_inset.set_title("Nigeria (context)", fontsize=8.5, fontweight="bold", pad=3)
    for spine in ax_inset.spines.values():
        spine.set_edgecolor("#334155")
        spine.set_linewidth(1.0)

    fig.suptitle("Geographical Extent of the Lagos State Study Area, Nigeria", fontsize=15, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    save_figure(fig, out_dir / "figure_3_1_study_area_lagos", dpi)


def figure_3_2_method_framework(out_dir: Path, dpi: int) -> None:
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    draw_box(
        ax,
        (0.03, 0.72),
        0.2,
        0.2,
        "#dbeafe",
        "Data Acquisition\nMODIS LST, NDVI, NDBI,\nMNDWI, WorldPop,\nLagos Boundary",
    )
    draw_box(
        ax,
        (0.27, 0.72),
        0.2,
        0.2,
        "#e0f2fe",
        "Preprocessing\nClipping, reprojection,\nannual aggregation,\nwater masking",
    )
    draw_box(
        ax,
        (0.51, 0.72),
        0.2,
        0.2,
        "#dcfce7",
        "Phase 1 (ESDA)\nMK-Sen trend,\ncorrelation,\nNDBI-LST quadrants",
    )
    draw_box(
        ax,
        (0.75, 0.72),
        0.2,
        0.2,
        "#fee2e2",
        "Phase 2 (TVDI + XGBoost)\nTVDI mapping, SHAP\nattribution, thermal\ndriver ranking",
    )

    draw_box(
        ax,
        (0.27, 0.35),
        0.2,
        0.2,
        "#fde68a",
        "Phase 3 (Forecasting)\nMLR and CNN-LSTM\n2030/2040 LST\nprojection maps",
    )
    draw_box(
        ax,
        (0.51, 0.35),
        0.2,
        0.2,
        "#f3e8ff",
        "Validation and Metrics\nRMSE, MAE, R2,\ntrend significance,\ncontribution percent",
    )
    draw_box(
        ax,
        (0.75, 0.35),
        0.2,
        0.2,
        "#fef9c3",
        "Decision Support Outputs\nHotspot maps,\nforecast risk surfaces,\nvegetation stress insights",
    )

    draw_arrow(ax, (0.23, 0.82), (0.27, 0.82))
    draw_arrow(ax, (0.47, 0.82), (0.51, 0.82))
    draw_arrow(ax, (0.71, 0.82), (0.75, 0.82))
    draw_arrow(ax, (0.61, 0.72), (0.37, 0.55))
    draw_arrow(ax, (0.37, 0.35), (0.51, 0.35))
    draw_arrow(ax, (0.71, 0.35), (0.75, 0.35))

    ax.text(
        0.5,
        0.95,
        "Methodological Framework for Geospatial Machine Learning Assessment",
        ha="center",
        va="center",
        fontsize=15,
        fontweight="bold",
    )

    save_figure(fig, out_dir / "figure_3_2_methodological_framework", dpi)


def figure_3_3_phase1_results(project_root: Path, out_dir: Path, dpi: int) -> None:
    """Figure 3.3 = Phase One results (Objectives i and ii): the actual LST
    trend and NDBI-LST relationship maps produced by this study's own
    analysis (not a placeholder/example)."""
    boundary = gpd.read_file(project_root / "data/vectors/lagos_boundary.geojson").to_crs("EPSG:32631")

    quad_arr, quad_extent, quad_crs = open_raster(project_root / "outputs/phase1/ndbi_lst_quadrant_map.tif")

    mk_ds = xr.open_dataset(project_root / "outputs/phase1/mk_sen_maps.nc")
    sen = mk_ds["sen_slope_c_per_year"].values.astype(float)
    pval = mk_ds["mk_pvalue"].values.astype(float)

    if "x" in mk_ds.coords and "y" in mk_ds.coords:
        xvals = mk_ds["x"].values
        yvals = mk_ds["y"].values
        sen_extent = (float(np.nanmin(xvals)), float(np.nanmax(xvals)), float(np.nanmin(yvals)), float(np.nanmax(yvals)))
    else:
        sen_extent = quad_extent

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7.4), constrained_layout=True)

    cmap_sen = plt.get_cmap("coolwarm")
    vabs = float(np.nanpercentile(np.abs(sen[np.isfinite(sen)]), 95)) if np.isfinite(sen).any() else 1.0
    im1 = ax1.imshow(sen, extent=sen_extent, origin="upper", cmap=cmap_sen, vmin=-vabs, vmax=vabs)
    boundary.boundary.plot(ax=ax1, color="black", linewidth=1.2)
    # geopandas' .plot() resets the axis back to equal aspect, so "auto"
    # must be (re-)applied AFTER every geopandas plot call, not before.
    ax1.set_aspect("auto")

    sig = np.where(pval < 0.05, 1.0, np.nan)
    cs = ax1.contourf(sig, levels=[0.5, 1.5], extent=sen_extent, colors="none", hatches=["////"])
    # Hatch patches are invisible unless their edge colour/width is set
    # explicitly (matplotlib does not draw a visible hatch on a fully
    # transparent fill by default). Handle both the pre- and post-3.8
    # contourf return types.
    try:
        hatch_artists = cs.collections
    except AttributeError:
        hatch_artists = [cs]
    for artist in hatch_artists:
        artist.set_edgecolor("#1f2937")
        artist.set_linewidth(0.0)

    ax1.set_title("(a) LST Trend, 2000-2024\n(Sen's Slope, °C/year)", fontweight="bold", fontsize=12)
    ax1.set_xlabel("Easting (m)")
    ax1.set_ylabel("Northing (m)")
    ax1.grid(alpha=0.15, linestyle=":")

    hatch_patch = mpatches.Patch(
        facecolor="white", edgecolor="#1f2937", hatch="////",
        label="Statistically significant\ntrend (Mann-Kendall,\np < 0.05)",
    )
    ax1.legend(handles=[hatch_patch], loc="lower left", frameon=True, fontsize=7.5)

    divider1 = make_axes_locatable(ax1)
    cax1 = divider1.append_axes("right", size="4%", pad=0.08)
    cb1 = fig.colorbar(im1, cax=cax1)
    cb1.set_label("Sen's slope (°C/year)")

    quad_colors = ["#93c5fd", "#fde68a", "#a7f3d0", "#fca5a5"]
    quad_cmap = matplotlib.colors.ListedColormap(quad_colors)
    quad_norm = matplotlib.colors.BoundaryNorm([0.5, 1.5, 2.5, 3.5, 4.5], quad_cmap.N)

    im2 = ax2.imshow(quad_arr, extent=quad_extent, origin="upper", cmap=quad_cmap, norm=quad_norm)
    boundary.boundary.plot(ax=ax2, color="black", linewidth=1.2)
    ax2.set_aspect("auto")
    ax2.set_title("(b) Built-Up Intensity vs. Surface\nHeat (NDBI-LST Quadrants)", fontweight="bold", fontsize=12)
    ax2.set_xlabel("Easting (m)")
    # No y-label here: ax2 sits immediately right of ax1's colorbar and
    # shares the same Northing axis, so a second "Northing (m)" label
    # would collide with the colorbar's own label under constrained_layout.
    ax2.grid(alpha=0.15, linestyle=":")

    labels = [
        "Low built-up / Low heat",
        "High built-up / Low heat",
        "Low built-up / High heat",
        "High built-up / High heat\n(critical UHI core)",
    ]
    handles = [mpatches.Patch(color=quad_colors[i], label=labels[i]) for i in range(4)]
    ax2.legend(handles=handles, loc="lower left", frameon=True, fontsize=7.5)

    fig.suptitle(
        "Historical LST Trend and the Built-Up/Heat Relationship, Lagos State (2000-2024)",
        fontsize=14,
        fontweight="bold",
    )

    _ = im2
    save_figure(fig, out_dir / "figure_3_3_phase1_trend_quadrant_overlay", dpi)


def figure_3_4_phase2_tvdi_shap(project_root: Path, out_dir: Path, dpi: int) -> None:
    """Phase 2 = TVDI + XGBoost + SHAP (Figure 3.4 / Figure 4.x)."""
    tvdi_path = project_root / "outputs/phase3/tvdi_2024.tif"
    tvdi, extent, _ = open_raster(tvdi_path)

    shap_df = pd.read_csv(project_root / "outputs/phase3/shap_feature_importance.csv")
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=True)

    ext_path = project_root / "outputs/phase3/shap_extreme_feature_contributions.csv"
    ext_df = pd.read_csv(ext_path)
    ext_df = ext_df.dropna(subset=["contribution_pct_extreme"])

    fig = plt.figure(figsize=(14, 6.8), constrained_layout=True)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.4, 1.0, 1.0])

    ax1 = fig.add_subplot(gs[0, 0])
    im = ax1.imshow(tvdi, extent=extent, origin="upper", cmap="YlOrRd", vmin=0, vmax=1)
    ax1.set_aspect("auto")
    ax1.set_title("(a) TVDI 2024", fontweight="bold")
    ax1.set_xlabel("Easting (m)")
    ax1.set_ylabel("Northing (m)")
    ax1.grid(alpha=0.12, linestyle=":")

    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="4%", pad=0.08)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("TVDI")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.barh(shap_df["feature"], shap_df["mean_abs_shap"], color="#60a5fa")
    ax2.set_title("(b) Mean |SHAP| Importance", fontweight="bold")
    ax2.set_xlabel("Mean absolute SHAP value")
    ax2.set_ylabel("Feature")
    ax2.grid(axis="x", alpha=0.2, linestyle=":")

    ax3 = fig.add_subplot(gs[0, 2])
    if ext_df.empty:
        ax3.text(0.5, 0.5, "No extreme TVDI subset\navailable", ha="center", va="center")
        ax3.set_axis_off()
    else:
        ext_df = ext_df.sort_values("contribution_pct_extreme", ascending=True)
        ax3.barh(ext_df["feature"], ext_df["contribution_pct_extreme"], color="#f59e0b")
        ax3.set_title("(c) Extreme TVDI\nContribution (%)", fontweight="bold")
        ax3.set_xlabel("Contribution percent")
        ax3.set_ylabel("Feature")
        ax3.grid(axis="x", alpha=0.2, linestyle=":")

    fig.suptitle("TVDI (2024) and SHAP-Based Attribution of Vegetation Stress Drivers, Lagos State", fontsize=14, fontweight="bold")

    save_figure(fig, out_dir / "figure_3_4_phase2_tvdi_shap", dpi)


def figure_3_5_phase3_forecasts(project_root: Path, out_dir: Path, dpi: int) -> None:
    """Phase 3 = MLR + CNN-LSTM-Attention forecasting (Figure 3.5 / Figure 4.x)."""
    paths = {
        "MLR 2030": project_root / "outputs/phase2_tf/mlr_lst_forecast_2030.tif",
        "MLR 2040": project_root / "outputs/phase2_tf/mlr_lst_forecast_2040.tif",
        "CNN-LSTM 2030": project_root / "outputs/phase2_tf/cnn_lstm_attention_lst_forecast_2030.tif",
        "CNN-LSTM 2040": project_root / "outputs/phase2_tf/cnn_lstm_attention_lst_forecast_2040.tif",
    }

    arrays: dict[str, np.ndarray] = {}
    extents: dict[str, tuple[float, float, float, float]] = {}
    for key, path in paths.items():
        arr, ext, _ = open_raster(path)
        arrays[key] = arr
        extents[key] = ext

    all_vals = np.concatenate([a[np.isfinite(a)] for a in arrays.values() if np.isfinite(a).any()])
    vmin = float(np.nanpercentile(all_vals, 2))
    vmax = float(np.nanpercentile(all_vals, 98))

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    cmap = plt.get_cmap("inferno")

    keys = list(paths.keys())
    for ax, key in zip(axes.ravel(), keys):
        im = ax.imshow(arrays[key], extent=extents[key], origin="upper", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_aspect("auto")
        ax.set_title(key, fontweight="bold")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.grid(alpha=0.12, linestyle=":")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label("Forecasted LST (C)")
    fig.suptitle("MLR and CNN-LSTM-Attention LST Forecasts for Lagos State (2030 and 2040)", fontsize=14, fontweight="bold")

    save_figure(fig, out_dir / "figure_3_5_phase3_lst_forecast_maps", dpi)


def main() -> None:
    args = parse_args()
    project_root = Path(args.project_root).resolve()
    out_dir = (project_root / args.output_dir).resolve()

    apply_style()

    figure_2_1_conceptual(out_dir, args.dpi)
    figure_3_1_study_area(project_root, out_dir, args.dpi)
    figure_3_2_method_framework(out_dir, args.dpi)
    figure_3_3_phase1_results(project_root, out_dir, args.dpi)
    figure_3_4_phase2_tvdi_shap(project_root, out_dir, args.dpi)
    figure_3_5_phase3_forecasts(project_root, out_dir, args.dpi)

    print("Publication-quality figures generated:")
    print(out_dir)


if __name__ == "__main__":
    main()
