#!/usr/bin/env python3
"""Download Phase 1 MODIS rasters for Lagos directly from Earth Engine to local folders.

Unlike Drive exports, this script fetches files via Earth Engine download URLs and saves
GeoTIFFs locally for immediate use in the ESDA pipeline.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import zipfile
from pathlib import Path

import ee
import requests


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct-download MODIS Phase 1 rasters from GEE")
    parser.add_argument(
        "--ee-project",
        default=None,
        help="Google Cloud project ID enabled for Earth Engine",
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--scale", type=int, default=1000)
    parser.add_argument("--crs", default="EPSG:32631")
    parser.add_argument("--timeout", type=int, default=600)

    parser.add_argument("--root-dir", default="data")
    parser.add_argument("--boundary-output", default="data/vectors/lagos_boundary.geojson")

    parser.add_argument("--lagos-fc", default="FAO/GAUL/2015/level1")
    parser.add_argument("--country-field", default="ADM0_NAME")
    parser.add_argument("--state-field", default="ADM1_NAME")
    parser.add_argument("--country-name", default="Nigeria")
    parser.add_argument("--state-name", default="Lagos")
    return parser.parse_args()


def init_ee(project: str | None) -> None:
    resolved_project = project or os.getenv("EE_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    try:
        ee.Initialize(project=resolved_project)
    except Exception as exc:
        msg = (
            "Earth Engine initialization failed. Run one of:\n"
            "  earthengine authenticate\n"
            "or\n"
            "  python -c \"import ee; ee.Authenticate()\"\n"
            "Then provide a project with one of:\n"
            "  --ee-project your-gcp-project-id\n"
            "  export EE_PROJECT=your-gcp-project-id"
        )
        raise RuntimeError(msg) from exc


def get_lagos_feature(args: argparse.Namespace) -> ee.Feature:
    fc = ee.FeatureCollection(args.lagos_fc)
    feat = fc.filter(ee.Filter.eq(args.country_field, args.country_name)).filter(
        ee.Filter.eq(args.state_field, args.state_name)
    )
    if feat.size().getInfo() == 0:
        raise ValueError("Could not find Lagos feature. Check boundary collection and field names.")
    return ee.Feature(feat.first())


def export_boundary_geojson(feature: ee.Feature, output_path: str) -> None:
    info = feature.getInfo()
    geojson = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": info.get("properties", {}),
                "geometry": info.get("geometry", {}),
            }
        ],
    }
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(geojson), encoding="utf-8")


def annual_modis_lst(year: int, region: ee.Geometry) -> ee.Image:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    return (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterDate(start, end)
        .select("LST_Day_1km")
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .rename("LST")
        .clip(region)
        .toFloat()
    )


def annual_modis_ndvi(year: int, region: ee.Geometry) -> ee.Image:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    return (
        ee.ImageCollection("MODIS/061/MOD13A2")
        .filterDate(start, end)
        .select("NDVI")
        .mean()
        .multiply(0.0001)
        .rename("NDVI")
        .clip(region)
        .toFloat()
    )


def annual_modis_ndbi_mndwi(year: int, region: ee.Geometry) -> tuple[ee.Image, ee.Image]:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")

    sr = ee.ImageCollection("MODIS/061/MOD09A1").filterDate(start, end).mean()
    nir = sr.select("sur_refl_b02").multiply(0.0001)
    swir1 = sr.select("sur_refl_b06").multiply(0.0001)
    green = sr.select("sur_refl_b04").multiply(0.0001)

    ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename("NDBI").clip(region).toFloat()
    mndwi = green.subtract(swir1).divide(green.add(swir1)).rename("MNDWI").clip(region).toFloat()
    return ndbi, mndwi


def download_geotiff(
    image: ee.Image,
    out_file: Path,
    region: ee.Geometry,
    scale: int,
    crs: str,
    timeout: int,
) -> None:
    out_file.parent.mkdir(parents=True, exist_ok=True)

    url = image.getDownloadURL(
        {
            "region": region,
            "scale": scale,
            "crs": crs,
            "format": "GEO_TIFF",
            "filePerBand": False,
        }
    )

    response = requests.get(url, timeout=timeout)
    response.raise_for_status()

    content_type = response.headers.get("Content-Type", "")
    payload = response.content

    if "application/zip" in content_type or payload[:2] == b"PK":
        with zipfile.ZipFile(io.BytesIO(payload)) as zf:
            tif_candidates = [n for n in zf.namelist() if n.lower().endswith(".tif")]
            if not tif_candidates:
                raise RuntimeError(f"No GeoTIFF found in archive for output {out_file}")
            with zf.open(tif_candidates[0]) as src, out_file.open("wb") as dst:
                dst.write(src.read())
    else:
        # Some responses may already return a direct GeoTIFF stream.
        out_file.write_bytes(payload)


def main() -> None:
    args = parse_args()
    init_ee(args.ee_project)

    root = Path(args.root_dir)
    lst_dir = root / "modis" / "lst"
    ndvi_dir = root / "indices" / "ndvi"
    ndbi_dir = root / "indices" / "ndbi"
    mndwi_dir = root / "indices" / "mndwi"

    lagos_feature = get_lagos_feature(args)
    region = lagos_feature.geometry()
    export_boundary_geojson(lagos_feature, args.boundary_output)

    for year in range(args.start_year, args.end_year + 1):
        date_tag = f"{year}-01-01"
        print(f"Downloading year {year}...")

        lst = annual_modis_lst(year, region)
        ndvi = annual_modis_ndvi(year, region)
        ndbi, mndwi = annual_modis_ndbi_mndwi(year, region)

        download_geotiff(
            image=lst,
            out_file=lst_dir / f"modis_lst_{date_tag}.tif",
            region=region,
            scale=args.scale,
            crs=args.crs,
            timeout=args.timeout,
        )
        download_geotiff(
            image=ndvi,
            out_file=ndvi_dir / f"modis_ndvi_{date_tag}.tif",
            region=region,
            scale=args.scale,
            crs=args.crs,
            timeout=args.timeout,
        )
        download_geotiff(
            image=ndbi,
            out_file=ndbi_dir / f"modis_ndbi_{date_tag}.tif",
            region=region,
            scale=args.scale,
            crs=args.crs,
            timeout=args.timeout,
        )
        download_geotiff(
            image=mndwi,
            out_file=mndwi_dir / f"modis_mndwi_{date_tag}.tif",
            region=region,
            scale=args.scale,
            crs=args.crs,
            timeout=args.timeout,
        )

    print("Done. Local Phase 1 rasters are ready.")
    print("Boundary file:", args.boundary_output)


if __name__ == "__main__":
    main()
