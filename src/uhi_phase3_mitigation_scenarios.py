#!/usr/bin/env python3
"""Phase 3b: Mitigation-scenario simulation for vegetation stress (TVDI).

This script quantifies the expected reduction in severe/extreme vegetation
thermal stress (TVDI) under three hypothetical mitigation interventions,
applied within the priority zone already identified by the earlier phases
of this pipeline:

    Priority zone = pixels that are simultaneously
        (a) in the "critical UHI core" from Phase 1
            (NDBI >= the Phase 1 median NDBI threshold, i.e. high built-up
            intensity), AND
        (b) at or above the severe-stress TVDI threshold in the most recent
            analysis year (default: 2024).

Three scenarios are simulated within that priority zone only; pixels
outside it are left at their observed baseline values:

    Scenario A - Greening only:
        NDVI is raised by a fixed increment (--delta-ndvi-greening) to
        represent feasible urban canopy/green-infrastructure expansion.
        The resulting LST change is estimated from a clean univariate
        LST ~ NDVI regression fitted directly on this study's own Lagos
        pixels for the analysis year (NOT the multi-predictor Phase 2 MLR
        coefficients, which are confounded by NDBI-NDVI collinearity and
        are therefore not appropriate for a causal what-if simulation).

    Scenario B - Reflective/cool pavement only:
        A fixed direct LST reduction (--lst-offset-pavement) is applied,
        representing the albedo-driven cooling reported in the cool-
        pavement literature (e.g. Vizzari, Leiva-Padilla & Hu, 2025,
        IntechOpen, DOI 10.5772/intechopen.1012755; Santamouris-line cool-
        pavement field studies more broadly report ~1.5-4 degC surface
        cooling). NDVI is left unchanged, since reflective pavement does
        not alter vegetation cover.

    Scenario C - Combined:
        Both perturbations are applied together to the same priority-zone
        pixels.

For every scenario (plus the observed baseline), TVDI is recomputed pixel-
by-pixel using the SAME empirical dry/wet edge coefficients already fitted
for that year in Phase 3 (outputs/phase3/tvdi_dry_wet_edges_by_year.csv),
so the only thing that changes between scenarios is the LST and/or NDVI
value fed into that existing, validated TVDI formula. Severe+extreme
stress area (TVDI >= --stress-threshold) is then tabulated for each
scenario, in both pixel count and km^2 (derived from the raster's actual
resolution), together with the percentage-point reduction relative to the
untreated baseline and relative to each single-lever scenario.

IMPORTANT - what this script is and is not:
This is a scenario / what-if simulation grounded in this study's own
empirical LST-NDVI relationship and in published cool-pavement cooling
magnitudes. It is NOT a physically-resolved urban microclimate model (e.g.
ENVI-met, SOLWEIG) and does NOT constitute field validation of actual
intervention performance in Lagos. The delta-NDVI and pavement cooling
assumptions are explicit, documented CLI parameters, not measured
outcomes, and are written verbatim into the output metadata file for full
transparency and reproducibility. Report results as "simulated" and
"literature-informed", never as measured or field-validated.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401
import xarray as xr
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap

matplotlib.use("Agg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 3b mitigation-scenario simulation")
    parser.add_argument("--lagos-boundary", required=True)
    parser.add_argument("--modis-lst-pattern", required=True)
    parser.add_argument("--ndvi-pattern", required=True)
    parser.add_argument("--ndbi-pattern", required=True)

    parser.add_argument(
        "--tvdi-raster",
        required=True,
        help="Path to the Phase 3 TVDI GeoTIFF for --scenario-year, e.g. outputs/phase3/tvdi_2024.tif",
    )
    parser.add_argument(
        "--tvdi-edges-csv",
        required=True,
        help="Path to outputs/phase3/tvdi_dry_wet_edges_by_year.csv (Phase 3 output)",
    )
    parser.add_argument(
        "--ndbi-quadrant-stats-csv",
        required=True,
        help="Path to outputs/phase1/ndbi_lst_quadrant_stats.csv (Phase 1 output), "
        "used to source the median NDBI threshold that defines the critical UHI core",
    )

    parser.add_argument("--scenario-year", type=int, default=2024)
    parser.add_argument("--stress-threshold", type=float, default=0.6, help="TVDI value at/above which a pixel is severe/extreme stress")
    parser.add_argument("--delta-ndvi-greening", type=float, default=0.15, help="Assumed feasible NDVI increase under urban greening (Scenario A)")
    parser.add_argument("--lst-offset-pavement", type=float, default=2.0, help="Assumed direct LST reduction in degC from reflective/cool pavement (Scenario B)")

    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--figure-output-dir",
        default="outputs/final/figures",
        help="Directory to also save figure-numbered copies of the map (Figure 4.10) and "
        "comparison chart (Figure 4.11), alongside all the other final report figures.",
    )
    parser.add_argument("--dpi", type=int, default=400, help="PNG export DPI for the scenario map figure")
    return parser.parse_args()


def load_single_year_raster(pattern: str, year: int) -> xr.DataArray:
    candidates = [p for p in sorted(Path().glob(pattern)) if str(year) in p.stem]
    if not candidates:
        raise FileNotFoundError(f"No file matching year {year} for pattern: {pattern}")
    da = xr.open_dataarray(candidates[0])
    if "band" in da.dims and da.sizes["band"] == 1:
        da = da.squeeze("band", drop=True)
    return da


def clip_to_boundary(da: xr.DataArray, boundary: gpd.GeoDataFrame) -> xr.DataArray:
    target = boundary
    if target.crs != da.rio.crs:
        target = target.to_crs(da.rio.crs)
    return da.rio.clip(target.geometry, target.crs, drop=True)


def fit_lst_ndvi_slope(lst: np.ndarray, ndvi: np.ndarray) -> tuple[float, float]:
    """Clean univariate OLS of LST on NDVI (this study's own pixels, one year).

    Deliberately univariate: the Phase 2 multiple-linear-regression NDVI
    coefficient is confounded by strong NDBI-NDVI collinearity (see
    docs/final_project_report.md, Phase 2 interpretation note) and must not
    be used for a causal what-if simulation. This slope is used only to
    project the LST response to a hypothetical NDVI increase, restricted to
    the priority zone.
    """
    valid = np.isfinite(lst) & np.isfinite(ndvi)
    slope, intercept = np.polyfit(ndvi[valid], lst[valid], 1)
    return float(slope), float(intercept)


def recompute_tvdi(lst: np.ndarray, ndvi: np.ndarray, a_dry: float, b_dry: float, a_wet: float, b_wet: float) -> np.ndarray:
    lst_max = a_dry + b_dry * ndvi
    lst_min = a_wet + b_wet * ndvi
    denom = lst_max - lst_min
    tvdi = (lst - lst_min) / denom
    return np.where(np.isfinite(tvdi), np.clip(tvdi, 0, 1), np.nan)


def stress_area_km2(tvdi: np.ndarray, threshold: float, pixel_area_km2: float) -> tuple[int, float]:
    n_pixels = int(np.nansum(tvdi >= threshold))
    return n_pixels, n_pixels * pixel_area_km2


def add_north_arrow(ax: plt.Axes, x: float = 0.92, y: float = 0.90) -> None:
    ax.annotate(
        "N", xy=(x, y), xytext=(x, y - 0.09), xycoords="axes fraction",
        ha="center", va="center", fontsize=9, fontweight="bold",
        arrowprops=dict(arrowstyle="-|>", lw=1.2, color="black"),
    )


def add_scale_bar(ax: plt.Axes) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    width, height = xmax - xmin, ymax - ymin
    raw_km = (width * 0.28) / 1000.0
    magnitude = 10 ** np.floor(np.log10(raw_km)) if raw_km > 0 else 1.0
    length_km = next((m * magnitude for m in (1, 2, 5, 10) if m * magnitude >= raw_km), 10 * magnitude)
    length_m = length_km * 1000.0
    x0, y0 = xmin + 0.05 * width, ymin + 0.06 * height
    ax.plot([x0, x0 + length_m], [y0, y0], color="black", lw=2.5, solid_capstyle="butt")
    ax.plot([x0, x0], [y0 - 0.008 * height, y0 + 0.008 * height], color="black", lw=1)
    ax.plot([x0 + length_m, x0 + length_m], [y0 - 0.008 * height, y0 + 0.008 * height], color="black", lw=1)
    ax.text(x0 + 0.5 * length_m, y0 + 0.015 * height, f"{length_km:g} km", ha="center", va="bottom", fontsize=7)


def save_scenario_maps(
    lst: xr.DataArray,
    tvdi_baseline: np.ndarray,
    tvdi_combined: np.ndarray,
    priority_mask: np.ndarray,
    boundary: gpd.GeoDataFrame,
    out_dir: Path,
    figure_dir: Path,
    scenario_year: int,
    dpi: int,
) -> None:
    """Publication-quality spatial comparison: baseline TVDI, Scenario C TVDI,
    and the priority mitigation zone, so the simulation is a map, not just a
    bar chart and a CSV.
    """
    bounds = lst.rio.bounds()
    extent = (bounds[0], bounds[2], bounds[1], bounds[3])
    boundary_proj = boundary.to_crs(lst.rio.crs)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5), constrained_layout=True)

    im0 = axes[0].imshow(tvdi_baseline, extent=extent, origin="upper", cmap="YlOrRd", vmin=0, vmax=1)
    boundary_proj.boundary.plot(ax=axes[0], color="black", linewidth=1.0)
    axes[0].set_title(f"(a) Baseline TVDI ({scenario_year})", fontweight="bold")

    im1 = axes[1].imshow(tvdi_combined, extent=extent, origin="upper", cmap="YlOrRd", vmin=0, vmax=1)
    boundary_proj.boundary.plot(ax=axes[1], color="black", linewidth=1.0)
    axes[1].set_title("(b) Scenario C: Combined Mitigation", fontweight="bold")

    priority_display = np.where(priority_mask, 1.0, np.nan)
    axes[2].imshow(tvdi_baseline, extent=extent, origin="upper", cmap="Greys", vmin=0, vmax=1, alpha=0.35)
    axes[2].imshow(priority_display, extent=extent, origin="upper", cmap=ListedColormap(["#d62728"]), vmin=0, vmax=1)
    boundary_proj.boundary.plot(ax=axes[2], color="black", linewidth=1.0)
    axes[2].set_title("(c) Priority Mitigation Zone", fontweight="bold")

    for ax in axes:
        ax.set_aspect("auto")
        ax.set_xlabel("Easting (m)")
        ax.set_ylabel("Northing (m)")
        ax.grid(alpha=0.12, linestyle=":")
        add_north_arrow(ax)
        add_scale_bar(ax)

    cbar = fig.colorbar(im1, ax=[axes[0], axes[1]], fraction=0.025, pad=0.02)
    cbar.set_label("TVDI")
    _ = im0

    fig.suptitle(
        f"Simulated Mitigation Impact on Vegetation Stress (TVDI), Priority Zone, {scenario_year}",
        fontsize=13,
        fontweight="bold",
    )

    out_base = out_dir / "mitigation_scenario_tvdi_maps"
    fig.savefig(out_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_base.with_suffix(".svg"), bbox_inches="tight")

    # Also drop a figure-numbered copy alongside all the other final report
    # figures (Figure 4.10), so everything lands in one place.
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig_base = figure_dir / "figure_4_10_mitigation_scenario_tvdi_maps"
    fig.savefig(fig_base.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(fig_base.with_suffix(".svg"), bbox_inches="tight")

    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_dir = Path(args.figure_output_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)

    boundary = gpd.read_file(args.lagos_boundary)

    lst = clip_to_boundary(load_single_year_raster(args.modis_lst_pattern, args.scenario_year), boundary)
    ndvi = clip_to_boundary(load_single_year_raster(args.ndvi_pattern, args.scenario_year), boundary)
    ndbi = clip_to_boundary(load_single_year_raster(args.ndbi_pattern, args.scenario_year), boundary)
    tvdi_observed = xr.open_dataarray(args.tvdi_raster)
    if "band" in tvdi_observed.dims and tvdi_observed.sizes["band"] == 1:
        tvdi_observed = tvdi_observed.squeeze("band", drop=True)

    ndvi = ndvi.rio.reproject_match(lst)
    ndbi = ndbi.rio.reproject_match(lst)
    tvdi_observed = tvdi_observed.rio.reproject_match(lst)

    lst_v, ndvi_v, ndbi_v, tvdi_v = lst.values, ndvi.values, ndbi.values, tvdi_observed.values

    # Empirical TVDI edge coefficients already fitted in Phase 3 for this year.
    edges = pd.read_csv(args.tvdi_edges_csv)
    row = edges.loc[edges["year"] == args.scenario_year].iloc[0]
    a_dry, b_dry, a_wet, b_wet = row["a_dry"], row["b_dry"], row["a_wet"], row["b_wet"]

    # Priority zone: Phase 1 critical UHI core (NDBI >= median threshold) AND
    # currently severe/extreme TVDI stress.
    quad = pd.read_csv(args.ndbi_quadrant_stats_csv)
    ndbi_threshold = float(quad["ndbi_threshold"].iloc[0])
    priority_mask = (ndbi_v >= ndbi_threshold) & (tvdi_v >= args.stress_threshold)

    x_res, y_res = lst.rio.resolution()
    pixel_area_km2 = abs(x_res * y_res) / 1e6

    # Scenario A: greening only.
    slope, intercept = fit_lst_ndvi_slope(lst_v, ndvi_v)
    ndvi_a = ndvi_v.copy()
    ndvi_a[priority_mask] = np.clip(ndvi_a[priority_mask] + args.delta_ndvi_greening, -1.0, 1.0)
    lst_a = lst_v.copy()
    lst_a[priority_mask] = lst_v[priority_mask] + slope * (ndvi_a[priority_mask] - ndvi_v[priority_mask])
    tvdi_a = recompute_tvdi(lst_a, ndvi_a, a_dry, b_dry, a_wet, b_wet)

    # Scenario B: reflective/cool pavement only.
    lst_b = lst_v.copy()
    lst_b[priority_mask] = lst_v[priority_mask] - args.lst_offset_pavement
    tvdi_b = recompute_tvdi(lst_b, ndvi_v, a_dry, b_dry, a_wet, b_wet)

    # Scenario C: combined.
    ndvi_c = ndvi_a
    lst_c = lst_v.copy()
    lst_c[priority_mask] = lst_a[priority_mask] - args.lst_offset_pavement
    tvdi_c = recompute_tvdi(lst_c, ndvi_c, a_dry, b_dry, a_wet, b_wet)

    scenarios = {
        "Baseline (observed)": tvdi_v,
        "Scenario A: Greening only": tvdi_a,
        "Scenario B: Reflective pavement only": tvdi_b,
        "Scenario C: Combined": tvdi_c,
    }

    baseline_pixels, baseline_km2 = stress_area_km2(tvdi_v, args.stress_threshold, pixel_area_km2)

    rows = []
    for name, arr in scenarios.items():
        n_pix, km2 = stress_area_km2(arr, args.stress_threshold, pixel_area_km2)
        pct_reduction_vs_baseline = 100.0 * (baseline_km2 - km2) / baseline_km2 if baseline_km2 > 0 else np.nan
        rows.append(
            {
                "scenario": name,
                "severe_extreme_stress_pixels": n_pix,
                "severe_extreme_stress_km2": km2,
                "pct_reduction_vs_baseline": pct_reduction_vs_baseline,
            }
        )
    results = pd.DataFrame(rows)

    # Combined vs. best single-lever advantage, in percentage points.
    best_single = max(
        results.loc[results["scenario"] == "Scenario A: Greening only", "pct_reduction_vs_baseline"].iloc[0],
        results.loc[results["scenario"] == "Scenario B: Reflective pavement only", "pct_reduction_vs_baseline"].iloc[0],
    )
    combined_pct = results.loc[results["scenario"] == "Scenario C: Combined", "pct_reduction_vs_baseline"].iloc[0]
    results["pct_points_combined_over_best_single"] = np.where(
        results["scenario"] == "Scenario C: Combined", combined_pct - best_single, np.nan
    )

    results.to_csv(out_dir / "mitigation_scenario_results.csv", index=False)

    metadata = {
        "scenario_year": args.scenario_year,
        "stress_threshold": args.stress_threshold,
        "priority_zone_definition": "NDBI >= Phase 1 median NDBI threshold AND TVDI >= stress_threshold",
        "ndbi_threshold_used": ndbi_threshold,
        "priority_zone_pixel_count": int(np.nansum(priority_mask)),
        "delta_ndvi_greening": args.delta_ndvi_greening,
        "lst_offset_pavement_degC": args.lst_offset_pavement,
        "lst_ndvi_slope_fitted_this_year": slope,
        "lst_ndvi_intercept_fitted_this_year": intercept,
        "pixel_area_km2": pixel_area_km2,
        "assumption_sources": {
            "delta_ndvi_greening": "Scenario assumption (feasible urban canopy/green-infrastructure expansion); not a measured intervention outcome.",
            "lst_offset_pavement": "Literature-informed (Vizzari, Leiva-Padilla & Hu, 2025, IntechOpen, DOI 10.5772/intechopen.1012755; general cool-pavement field literature reports ~1.5-4 degC surface cooling); not measured in Lagos.",
        },
        "caveat": "This is a scenario simulation layered on the study's own fitted TVDI edges, not a physically-resolved microclimate model and not field-validated for Lagos.",
    }
    (out_dir / "mitigation_scenario_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    # Comparison figure.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(results["scenario"], results["severe_extreme_stress_km2"], color=["#888888", "#4CAF50", "#2196F3", "#8E44AD"])
    ax.set_ylabel("Severe + extreme stress area (km$^2$)")
    ax.set_title(f"Simulated mitigation scenarios, priority zone, {args.scenario_year}")
    plt.xticks(rotation=20, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "mitigation_scenario_comparison.png", dpi=300)

    # Also drop a figure-numbered copy alongside all the other final report
    # figures (Figure 4.11), so everything lands in one place.
    fig.savefig(figure_dir / "figure_4_11_severe_stress_area_by_scenario.png", dpi=args.dpi, bbox_inches="tight")
    fig.savefig(figure_dir / "figure_4_11_severe_stress_area_by_scenario.svg", bbox_inches="tight")
    plt.close(fig)

    save_scenario_maps(lst, tvdi_v, tvdi_c, priority_mask, boundary, out_dir, figure_dir, args.scenario_year, args.dpi)

    print("Mitigation scenario simulation complete.")
    print(results.to_string(index=False))
    print(f"Outputs written to: {out_dir.resolve()}")
    print(f"Figure-numbered copies also written to: {figure_dir.resolve()}")


if __name__ == "__main__":
    main()
