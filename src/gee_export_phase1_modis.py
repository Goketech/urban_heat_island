#!/usr/bin/env python3
"""Create Google Earth Engine export tasks for Phase 1 MODIS products over Lagos.

Exports one GeoTIFF per year for:
- LST (Celsius) from MOD11A1 LST_Day_1km
- NDVI (unitless) from MOD13A2 NDVI
- NDBI (unitless) from MOD09A1 NIR/SWIR1
- MNDWI (unitless) from MOD09A1 Green/SWIR1

All exports are clipped to Lagos and generated at a common target scale
(default 1000 m) so Phase 1 ESDA rasters align out of the box.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import ee


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Queue GEE export tasks for Phase 1 MODIS data")
    parser.add_argument(
        "--ee-project",
        default=None,
        help="Google Cloud project ID enabled for Earth Engine",
    )
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--drive-folder", default="UHI_Lagos_Phase1")
    parser.add_argument("--scale", type=int, default=1000, help="Export scale in meters")
    parser.add_argument("--crs", default="EPSG:32631")
    parser.add_argument("--max-pixels", type=int, default=1_000_000_000)
    parser.add_argument(
        "--boundary-output",
        default="data/vectors/lagos_boundary.geojson",
        help="Local GeoJSON path for Lagos boundary copy",
    )
    parser.add_argument(
        "--lagos-fc",
        default="FAO/GAUL/2015/level1",
        help="FeatureCollection containing state/province features",
    )
    parser.add_argument("--country-field", default="ADM0_NAME")
    parser.add_argument("--state-field", default="ADM1_NAME")
    parser.add_argument("--country-name", default="Nigeria")
    parser.add_argument("--state-name", default="Lagos")
    return parser.parse_args()


def init_ee(project: str | None) -> None:
    resolved_project = project or os.getenv("EE_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    try:
        ee.Initialize(project=resolved_project)
    except Exception as exc:  # pragma: no cover
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
    count = feat.size().getInfo()
    if count == 0:
        raise ValueError(
            "No Lagos feature found. Check --lagos-fc and field/value arguments for the boundary dataset."
        )
    return ee.Feature(feat.first())


def export_boundary_geojson(feature: ee.Feature, output_path: str) -> None:
    # Small boundary export using client-side GeoJSON for reproducibility in local workflows.
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


def _start_export(
    image: ee.Image,
    description: str,
    file_name_prefix: str,
    folder: str,
    region: ee.Geometry,
    scale: int,
    crs: str,
    max_pixels: int,
) -> None:
    task = ee.batch.Export.image.toDrive(
        image=image,
        description=description,
        folder=folder,
        fileNamePrefix=file_name_prefix,
        region=region,
        scale=scale,
        crs=crs,
        maxPixels=max_pixels,
        fileFormat="GeoTIFF",
    )
    task.start()


def annual_modis_lst(year: int, region: ee.Geometry) -> ee.Image:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    lst = (
        ee.ImageCollection("MODIS/061/MOD11A1")
        .filterDate(start, end)
        .select("LST_Day_1km")
        .mean()
        .multiply(0.02)
        .subtract(273.15)
        .rename("LST")
    )
    return lst.clip(region)


def annual_modis_ndvi(year: int, region: ee.Geometry) -> ee.Image:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")
    ndvi = (
        ee.ImageCollection("MODIS/061/MOD13A2")
        .filterDate(start, end)
        .select("NDVI")
        .mean()
        .multiply(0.0001)
        .rename("NDVI")
    )
    return ndvi.clip(region)


def annual_modis_ndbi_mndwi(year: int, region: ee.Geometry) -> tuple[ee.Image, ee.Image]:
    start = ee.Date.fromYMD(year, 1, 1)
    end = start.advance(1, "year")

    sr = ee.ImageCollection("MODIS/061/MOD09A1").filterDate(start, end).mean()
    nir = sr.select("sur_refl_b02").multiply(0.0001)
    swir1 = sr.select("sur_refl_b06").multiply(0.0001)
    green = sr.select("sur_refl_b04").multiply(0.0001)

    ndbi = swir1.subtract(nir).divide(swir1.add(nir)).rename("NDBI").clip(region)
    mndwi = green.subtract(swir1).divide(green.add(swir1)).rename("MNDWI").clip(region)
    return ndbi, mndwi


def queue_exports(args: argparse.Namespace, region: ee.Geometry) -> None:
    for year in range(args.start_year, args.end_year + 1):
        date_tag = f"{year}-01-01"

        lst = annual_modis_lst(year, region)
        ndvi = annual_modis_ndvi(year, region)
        ndbi, mndwi = annual_modis_ndbi_mndwi(year, region)

        _start_export(
            image=lst,
            description=f"lagos_modis_lst_{year}",
            file_name_prefix=f"modis_lst_{date_tag}",
            folder=args.drive_folder,
            region=region,
            scale=args.scale,
            crs=args.crs,
            max_pixels=args.max_pixels,
        )
        _start_export(
            image=ndvi,
            description=f"lagos_modis_ndvi_{year}",
            file_name_prefix=f"modis_ndvi_{date_tag}",
            folder=args.drive_folder,
            region=region,
            scale=args.scale,
            crs=args.crs,
            max_pixels=args.max_pixels,
        )
        _start_export(
            image=ndbi,
            description=f"lagos_modis_ndbi_{year}",
            file_name_prefix=f"modis_ndbi_{date_tag}",
            folder=args.drive_folder,
            region=region,
            scale=args.scale,
            crs=args.crs,
            max_pixels=args.max_pixels,
        )
        _start_export(
            image=mndwi,
            description=f"lagos_modis_mndwi_{year}",
            file_name_prefix=f"modis_mndwi_{date_tag}",
            folder=args.drive_folder,
            region=region,
            scale=args.scale,
            crs=args.crs,
            max_pixels=args.max_pixels,
        )


def main() -> None:
    args = parse_args()
    init_ee(args.ee_project)

    lagos_feature = get_lagos_feature(args)
    region = lagos_feature.geometry()

    export_boundary_geojson(lagos_feature, args.boundary_output)
    queue_exports(args, region)

    years = args.end_year - args.start_year + 1
    task_count = years * 4
    print("Boundary saved to:", args.boundary_output)
    print(f"Queued {task_count} GEE export tasks in Drive folder '{args.drive_folder}'.")
    print("Open https://code.earthengine.google.com/tasks or use the GEE Python task list to monitor progress.")


if __name__ == "__main__":
    main()
