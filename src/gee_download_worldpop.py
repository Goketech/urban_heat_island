#!/usr/bin/env python3
"""Download annual WorldPop population rasters for Lagos from Earth Engine."""

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
    parser = argparse.ArgumentParser(description="Download WorldPop annual population for Lagos")
    parser.add_argument("--ee-project", default=None, help="Google Cloud project ID for Earth Engine")
    parser.add_argument("--start-year", type=int, default=2000)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--worldpop-country-code", default="NGA", help="ISO3 country code used in WorldPop collection")
    parser.add_argument("--scale", type=int, default=1000)
    parser.add_argument("--crs", default="EPSG:32631")
    parser.add_argument("--timeout", type=int, default=1200)
    parser.add_argument("--output-dir", default="data/worldpop")
    parser.add_argument("--boundary-output", default="data/vectors/lagos_boundary.geojson")

    parser.add_argument("--lagos-fc", default="FAO/GAUL/2015/level1")
    parser.add_argument("--country-field", default="ADM0_NAME")
    parser.add_argument("--state-field", default="ADM1_NAME")
    parser.add_argument("--country-name", default="Nigeria")
    parser.add_argument("--state-name", default="Lagos")
    return parser.parse_args()


def init_ee(project: str | None) -> None:
    resolved_project = project or os.getenv("EE_PROJECT") or os.getenv("GOOGLE_CLOUD_PROJECT")
    ee.Initialize(project=resolved_project)


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


def worldpop_year_image(year: int, region: ee.Geometry, country_code: str) -> ee.Image:
    col = (
        ee.ImageCollection("WorldPop/GP/100m/pop")
        .filter(ee.Filter.eq("country", country_code))
        .filter(ee.Filter.eq("year", year))
    )
    img = ee.Image(col.first())
    # Keep names predictable for local pipeline.
    return img.select("population").rename("POP_DENS").clip(region).toFloat()


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
        out_file.write_bytes(payload)


def main() -> None:
    args = parse_args()
    init_ee(args.ee_project)

    lagos_feature = get_lagos_feature(args)
    region = lagos_feature.geometry()

    export_boundary_geojson(lagos_feature, args.boundary_output)

    for year in range(args.start_year, args.end_year + 1):
        print(f"Downloading WorldPop {year}...")
        image = worldpop_year_image(year, region, country_code=args.worldpop_country_code)
        out_path = Path(args.output_dir) / f"worldpop_popdens_{year}-01-01.tif"
        download_geotiff(
            image=image,
            out_file=out_path,
            region=region,
            scale=args.scale,
            crs=args.crs,
            timeout=args.timeout,
        )

    print(f"Done. WorldPop rasters written to: {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
