#!/usr/bin/env python3
"""Phase 1 ESDA pipeline for Lagos UHI analysis.

This script computes:
1) Pixelwise Mann-Kendall trend test and Sen's slope on MODIS LST annual means.
2) Spatial bivariate correlations: NDBI-LST and NDVI-LST.
3) NDBI-LST quadrant analysis (Low/Low, High/Low, Low/High, High/High).

Inputs are expected as raster files readable by xarray/rioxarray (GeoTIFF/NetCDF),
with either a `time` dimension or stack that can be reduced via mean.
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple

import geopandas as gpd
import numpy as np
import pandas as pd
import rioxarray  # noqa: F401  # required to register the rio accessor
import xarray as xr
from scipy.stats import norm, pearsonr, spearmanr


@dataclass
class CorrelationResult:
    x_name: str
    y_name: str
    pearson_r: float
    pearson_p: float
    spearman_rho: float
    spearman_p: float
    n: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Phase 1 ESDA for Lagos UHI")
    parser.add_argument("--lagos-boundary", required=True, help="Path to Lagos boundary vector file")
    parser.add_argument("--modis-lst-pattern", required=True, help="Glob for MODIS LST rasters/NetCDF")
    parser.add_argument("--ndbi-pattern", required=True, help="Glob for NDBI rasters")
    parser.add_argument("--ndvi-pattern", required=True, help="Glob for NDVI rasters")
    parser.add_argument("--water-mask-pattern", default=None, help="Optional glob for MNDWI rasters")
    parser.add_argument("--output-dir", required=True, help="Output directory")

    parser.add_argument(
        "--lst-scale-factor",
        type=float,
        default=0.02,
        help="Scale factor for raw MODIS LST values",
    )
    parser.add_argument(
        "--lst-offset-celsius",
        type=float,
        default=-273.15,
        help="Offset after scaling to convert to Celsius",
    )
    parser.add_argument(
        "--min-years",
        type=int,
        default=8,
        help="Minimum yearly observations needed for MK/Sen computations",
    )
    parser.add_argument(
        "--quadrant-threshold-mode",
        choices=["median", "mean"],
        default="median",
        help="Threshold function for low/high quadrants",
    )
    parser.add_argument(
        "--critical-alpha",
        type=float,
        default=0.05,
        help="Significance level for Mann-Kendall trend",
    )
    return parser.parse_args()


def find_files(pattern: str) -> list[Path]:
    files = sorted(Path().glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {pattern}")
    return files


def load_polygon(path: str) -> gpd.GeoDataFrame:
    gdf = gpd.read_file(path)
    if gdf.empty:
        raise ValueError("Lagos boundary file is empty.")
    return gdf


def ensure_time_dimension(da: xr.DataArray, files: list[Path]) -> xr.DataArray:
    if "time" in da.dims:
        return da
    # If no explicit time exists, build a synthetic time axis from file order.
    time_index = pd.date_range("2000-01-01", periods=da.sizes.get("band", len(files)), freq="MS")
    if "band" in da.dims and da.sizes["band"] == len(time_index):
        da = da.rename({"band": "time"}).assign_coords(time=time_index)
        return da
    raise ValueError("Unable to infer time dimension from input stack.")


def read_raster_stack(pattern: str, variable_name: str) -> xr.DataArray:
    files = find_files(pattern)
    if len(files) == 1:
        path = files[0]
        if path.suffix.lower() in {".nc", ".nc4"}:
            ds = xr.open_dataset(path)
            if variable_name in ds:
                da = ds[variable_name]
            else:
                # fallback: first data variable
                da = ds[list(ds.data_vars)[0]]
        else:
            da = xr.open_dataarray(path)
            if da.ndim == 2:
                da = da.expand_dims(time=[pd.Timestamp("2000-01-01")])
        return ensure_time_dimension(da, files)

    arrays = []
    timestamps = []

    def _parse_timestamp_from_stem(stem: str) -> pd.Timestamp | pd.NaT:
        # Accept names like modis_lst_2000-01-01 or just 2000-01-01.
        match = re.search(r"(\d{4}-\d{2}-\d{2})", stem)
        if match:
            return pd.to_datetime(match.group(1), errors="coerce")
        return pd.to_datetime(stem, errors="coerce")

    for file in files:
        da = xr.open_dataarray(file)
        if "band" in da.dims and da.sizes["band"] == 1:
            da = da.squeeze("band", drop=True)
        arrays.append(da)
        timestamps.append(_parse_timestamp_from_stem(file.stem))

    # Replace parse failures with a regular index while preserving order.
    if any(pd.isna(ts) for ts in timestamps):
        timestamps = list(pd.date_range("2000-01-01", periods=len(files), freq="MS"))

    stack = xr.concat(arrays, dim="time").assign_coords(time=pd.DatetimeIndex(timestamps))
    return stack


def clip_to_boundary(da: xr.DataArray, boundary: gpd.GeoDataFrame) -> xr.DataArray:
    if not hasattr(da, "rio"):
        raise ValueError("Input DataArray missing rioxarray accessor. Ensure rioxarray is installed.")

    target = boundary
    if boundary.crs is None:
        raise ValueError("Boundary CRS is undefined. Please set a valid CRS in the boundary file.")

    if da.rio.crs is None:
        raise ValueError("Raster CRS is undefined. Please provide CRS-aware rasters.")

    if boundary.crs != da.rio.crs:
        target = boundary.to_crs(da.rio.crs)

    return da.rio.clip(target.geometry, target.crs, drop=True)


def mk_sen_1d(series: np.ndarray, min_years: int) -> Tuple[float, float, float, float]:
    """Return (tau, p_value, sen_slope, z_score)."""
    y = np.asarray(series, dtype=float)
    y = y[np.isfinite(y)]
    n = y.size
    if n < min_years:
        return np.nan, np.nan, np.nan, np.nan

    # Mann-Kendall S statistic.
    s = 0
    slopes = []
    for i in range(n - 1):
        diff = y[i + 1 :] - y[i]
        s += np.sign(diff).sum()
        slopes.extend((diff / np.arange(1, n - i)).tolist())

    # Variance of S with tie correction.
    unique, counts = np.unique(y, return_counts=True)
    tie_term = np.sum(counts * (counts - 1) * (2 * counts + 5))
    var_s = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if var_s <= 0:
        return np.nan, np.nan, np.nan, np.nan

    if s > 0:
        z = (s - 1) / np.sqrt(var_s)
    elif s < 0:
        z = (s + 1) / np.sqrt(var_s)
    else:
        z = 0.0

    p = 2.0 * (1.0 - norm.cdf(abs(z)))
    tau = s / (0.5 * n * (n - 1))
    sen = float(np.median(slopes)) if slopes else np.nan
    return float(tau), float(p), sen, float(z)


def compute_mk_sen_maps(lst_annual: xr.DataArray, min_years: int) -> xr.Dataset:
    stacked = lst_annual.stack(pixel=("y", "x"))
    values = stacked.values  # shape: time x pixel

    tau = np.full(values.shape[1], np.nan, dtype=float)
    pval = np.full(values.shape[1], np.nan, dtype=float)
    sen = np.full(values.shape[1], np.nan, dtype=float)
    zscr = np.full(values.shape[1], np.nan, dtype=float)

    for j in range(values.shape[1]):
        t, p, s, z = mk_sen_1d(values[:, j], min_years=min_years)
        tau[j] = t
        pval[j] = p
        sen[j] = s
        zscr[j] = z

    template = stacked.isel(time=0).drop_vars("time")
    ds = xr.Dataset(
        {
            "mk_tau": xr.DataArray(tau, coords=template.coords, dims=template.dims),
            "mk_pvalue": xr.DataArray(pval, coords=template.coords, dims=template.dims),
            "sen_slope_c_per_year": xr.DataArray(sen, coords=template.coords, dims=template.dims),
            "mk_zscore": xr.DataArray(zscr, coords=template.coords, dims=template.dims),
        }
    ).unstack("pixel")
    return ds


def flatten_valid(*arrays: Iterable[xr.DataArray]) -> np.ndarray:
    valid = None
    data = []
    for da in arrays:
        arr = np.asarray(da.values, dtype=float).ravel()
        data.append(arr)
        mask = np.isfinite(arr)
        valid = mask if valid is None else (valid & mask)

    return np.vstack([arr[valid] for arr in data])


def compute_correlations(
    lst_mean: xr.DataArray,
    ndbi_mean: xr.DataArray,
    ndvi_mean: xr.DataArray,
    water_mask: xr.DataArray | None,
) -> pd.DataFrame:
    lst = lst_mean
    ndbi = ndbi_mean
    ndvi = ndvi_mean

    if water_mask is not None:
        # Exclude water (MNDWI > 0) from analysis.
        non_water = xr.where(water_mask <= 0, 1.0, np.nan)
        lst = lst * non_water
        ndbi = ndbi * non_water
        ndvi = ndvi * non_water

    ndbi_al, lst_al = xr.align(ndbi, lst, join="inner")
    ndvi_al, lst_al2 = xr.align(ndvi, lst, join="inner")

    ndbi_vals, lst_vals = flatten_valid(ndbi_al, lst_al)
    ndvi_vals, lst_vals2 = flatten_valid(ndvi_al, lst_al2)

    stats = []
    for x_name, y_name, x, y in [
        ("NDBI", "LST", ndbi_vals, lst_vals),
        ("NDVI", "LST", ndvi_vals, lst_vals2),
    ]:
        pr, pp = pearsonr(x, y)
        sr, sp = spearmanr(x, y)
        stats.append(
            CorrelationResult(
                x_name=x_name,
                y_name=y_name,
                pearson_r=pr,
                pearson_p=pp,
                spearman_rho=sr,
                spearman_p=sp,
                n=int(x.size),
            )
        )

    return pd.DataFrame([s.__dict__ for s in stats])


def classify_quadrants(
    ndbi: xr.DataArray,
    lst: xr.DataArray,
    threshold_mode: str,
) -> tuple[pd.DataFrame, float, xr.DataArray]:
    ndbi_al, lst_al = xr.align(ndbi, lst, join="inner")
    x, y = flatten_valid(ndbi_al, lst_al)

    if threshold_mode == "median":
        tx = float(np.median(x))
        ty = float(np.median(y))
    else:
        tx = float(np.mean(x))
        ty = float(np.mean(y))

    # 1=Low/Low, 2=High/Low, 3=Low/High, 4=High/High
    quad = xr.full_like(ndbi_al, np.nan, dtype=float)
    quad = xr.where((ndbi_al < tx) & (lst_al < ty), 1, quad)
    quad = xr.where((ndbi_al >= tx) & (lst_al < ty), 2, quad)
    quad = xr.where((ndbi_al < tx) & (lst_al >= ty), 3, quad)
    quad = xr.where((ndbi_al >= tx) & (lst_al >= ty), 4, quad)

    arr = quad.values.ravel()
    arr = arr[np.isfinite(arr)]
    counts = pd.Series(arr).value_counts().sort_index()
    labels = {
        1.0: "Low NDBI / Low LST",
        2.0: "High NDBI / Low LST",
        3.0: "Low NDBI / High LST",
        4.0: "High NDBI / High LST",
    }

    total = int(counts.sum())
    rows = []
    for code, label in labels.items():
        c = int(counts.get(code, 0))
        rows.append(
            {
                "quadrant_code": int(code),
                "quadrant": label,
                "pixel_count": c,
                "percentage": (100.0 * c / total) if total > 0 else np.nan,
                "ndbi_threshold": tx,
                "lst_threshold": ty,
            }
        )

    table = pd.DataFrame(rows)
    critical_pct = float(table.loc[table["quadrant_code"] == 4, "percentage"].iloc[0])
    return table, critical_pct, quad


def summarize_mk_state(ds: xr.Dataset, alpha: float) -> pd.DataFrame:
    tau = ds["mk_tau"].values.ravel()
    p = ds["mk_pvalue"].values.ravel()
    sen = ds["sen_slope_c_per_year"].values.ravel()

    valid = np.isfinite(tau) & np.isfinite(p) & np.isfinite(sen)
    tau = tau[valid]
    p = p[valid]
    sen = sen[valid]

    inc_sig = int(((sen > 0) & (p < alpha)).sum())
    dec_sig = int(((sen < 0) & (p < alpha)).sum())
    nonsig = int((p >= alpha).sum())

    return pd.DataFrame(
        [
            {
                "n_valid_pixels": int(valid.sum()),
                "significant_increasing_pixels": inc_sig,
                "significant_decreasing_pixels": dec_sig,
                "non_significant_pixels": nonsig,
                "median_sen_slope_c_per_year": float(np.nanmedian(sen)) if sen.size else np.nan,
                "mean_sen_slope_c_per_year": float(np.nanmean(sen)) if sen.size else np.nan,
                "median_tau": float(np.nanmedian(tau)) if tau.size else np.nan,
                "alpha": alpha,
            }
        ]
    )


def build_markdown_summary(
    mk_state: pd.DataFrame,
    corr: pd.DataFrame,
    quad: pd.DataFrame,
    critical_pct: float,
) -> str:
    lines = []
    lines.append("# Phase 1 ESDA Results")
    lines.append("")
    lines.append("## Mann-Kendall + Sen's Slope (Lagos-wide summary)")
    lines.append(mk_state.to_markdown(index=False))
    lines.append("")
    lines.append("## Bivariate Spatial Correlations")
    lines.append(corr.to_markdown(index=False))
    lines.append("")
    lines.append("## NDBI-LST Quadrant Overlay")
    lines.append(quad.to_markdown(index=False))
    lines.append("")
    lines.append(f"Critical UHI core (High NDBI / High LST): **{critical_pct:.2f}%** of valid Lagos pixels.")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    boundary = load_polygon(args.lagos_boundary)

    lst = read_raster_stack(args.modis_lst_pattern, variable_name="LST")
    lst = clip_to_boundary(lst, boundary)
    lst_c = lst.astype("float32") * args.lst_scale_factor + args.lst_offset_celsius

    # Aggregate to yearly means for robust non-parametric trend analysis.
    lst_annual = lst_c.resample(time="YS").mean(skipna=True)

    mk_ds = compute_mk_sen_maps(lst_annual, min_years=args.min_years)
    mk_state = summarize_mk_state(mk_ds, alpha=args.critical_alpha)

    ndbi = read_raster_stack(args.ndbi_pattern, variable_name="NDBI")
    ndvi = read_raster_stack(args.ndvi_pattern, variable_name="NDVI")

    ndbi = clip_to_boundary(ndbi, boundary)
    ndvi = clip_to_boundary(ndvi, boundary)

    # Mean composites for spatial relationship analysis.
    ndbi_mean = ndbi.mean(dim="time", skipna=True)
    ndvi_mean = ndvi.mean(dim="time", skipna=True)
    lst_mean = lst_c.mean(dim="time", skipna=True)

    water_mask = None
    if args.water_mask_pattern:
        mndwi = read_raster_stack(args.water_mask_pattern, variable_name="MNDWI")
        mndwi = clip_to_boundary(mndwi, boundary)
        water_mask = mndwi.mean(dim="time", skipna=True)

    corr = compute_correlations(lst_mean, ndbi_mean, ndvi_mean, water_mask=water_mask)
    quad_table, critical_pct, quad_map = classify_quadrants(
        ndbi=ndbi_mean,
        lst=lst_mean,
        threshold_mode=args.quadrant_threshold_mode,
    )

    # Persist raw outputs.
    mk_ds.to_netcdf(output_dir / "mk_sen_maps.nc")
    mk_state.to_csv(output_dir / "mk_state_summary.csv", index=False)
    corr.to_csv(output_dir / "correlation_matrix.csv", index=False)
    quad_table.to_csv(output_dir / "ndbi_lst_quadrant_stats.csv", index=False)

    quad_map.name = "ndbi_lst_quadrant"
    quad_map.rio.write_crs(lst_c.rio.crs, inplace=True)
    quad_map.rio.to_raster(output_dir / "ndbi_lst_quadrant_map.tif")

    report = build_markdown_summary(mk_state, corr, quad_table, critical_pct)
    (output_dir / "phase1_summary.md").write_text(report, encoding="utf-8")

    print("Phase 1 ESDA complete.")
    print(f"Outputs written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
