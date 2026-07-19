# Urban Heat Island Analysis - Lagos (Heat Scars and Green Shields)

This repository currently contains the executable **Phase 1 (ESDA)** workflow for:
- MODIS LST pixelwise Mann-Kendall trend test
- Sen's slope trend magnitude
- Spatial bivariate correlations: NDBI-LST and NDVI-LST
- NDBI-LST quadrant overlay and critical UHI core percentage

It also includes a Google Earth Engine exporter to generate aligned Lagos rasters when raw files are not already available.

## Environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Get Data From Google Earth Engine (Phase 1 Inputs)

Authenticate once:

```bash
earthengine authenticate
```

Set the default Earth Engine project (required by current EE API versions):

```bash
earthengine set_project your-gcp-project-id
```

Or export it per shell session:

```bash
export EE_PROJECT=your-gcp-project-id
```

If browser auth fails in terminal-only environments, use Python auth:

```bash
python -c "import ee; ee.Authenticate(); ee.Initialize()"
```

Optional (service account auth):

```bash
export GOOGLE_APPLICATION_CREDENTIALS="/absolute/path/to/service-account-key.json"
```

Then initialize in Python with a project explicitly:

```python
import ee
ee.Initialize(project="your-gcp-project-id")
```

Queue annual exports for 2000-2024 (LST, NDVI, NDBI, MNDWI) and save Lagos boundary locally:

```bash
python src/gee_export_phase1_modis.py \
  --ee-project your-gcp-project-id \
  --start-year 2000 \
  --end-year 2024 \
  --drive-folder UHI_Lagos_Phase1 \
  --boundary-output data/vectors/lagos_boundary.geojson
```

This script creates Google Drive export tasks (it does not instantly download files). After tasks complete,
place downloaded GeoTIFFs into:

- `data/modis/lst/` for `modis_lst_YYYY-01-01.tif`
- `data/indices/ndvi/` for `modis_ndvi_YYYY-01-01.tif`
- `data/indices/ndbi/` for `modis_ndbi_YYYY-01-01.tif`
- `data/indices/mndwi/` for `modis_mndwi_YYYY-01-01.tif`

### Direct API Download (no manual Drive step)

Use this script to download yearly rasters directly into the project folders:

```bash
python src/gee_download_phase1_modis.py \
  --ee-project your-gcp-project-id \
  --start-year 2000 \
  --end-year 2024 \
  --root-dir data \
  --boundary-output data/vectors/lagos_boundary.geojson
```

This will write to:
- `data/modis/lst/`
- `data/indices/ndvi/`
- `data/indices/ndbi/`
- `data/indices/mndwi/`

If downloads time out on slower connections, rerun with a higher timeout:

```bash
python src/gee_download_phase1_modis.py --timeout 1800
```

## Data Inputs (expected)

- Lagos state boundary vector (GeoPackage/Shapefile)
- MODIS LST stack (MOD11A1-derived rasters or NetCDF)
- NDBI stack
- NDVI stack
- Optional MNDWI stack for water masking

All rasters must have valid CRS metadata and be readable by xarray/rioxarray.

## Run Phase 1

```bash
python src/uhi_phase1_esda.py \
  --lagos-boundary data/vectors/lagos_boundary.gpkg \
  --modis-lst-pattern "data/modis/lst/*.tif" \
  --ndbi-pattern "data/indices/ndbi/*.tif" \
  --ndvi-pattern "data/indices/ndvi/*.tif" \
  --water-mask-pattern "data/indices/mndwi/*.tif" \
  --lst-scale-factor 1.0 \
  --lst-offset-celsius 0.0 \
  --output-dir outputs/phase1
```

Note: the direct GEE downloader already converts MODIS LST to Celsius. Keep
`--lst-scale-factor 1.0` and `--lst-offset-celsius 0.0` for those files.

## Phase 1 Outputs

- `mk_sen_maps.nc`
- `mk_state_summary.csv`
- `correlation_matrix.csv`
- `ndbi_lst_quadrant_stats.csv`
- `ndbi_lst_quadrant_map.tif`
- `phase1_summary.md`

## Baseline Documentation

See `docs/phase1_baseline_documentation.md` for:
- Band harmonization assumptions (Landsat/Sentinel-2)
- Strict index equations (NDVI, NDBI, LST, TVDI)
- Preprocessing constraints and hypothesis framing
