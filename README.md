# Ocean Current Retrieval from Sentinel-1 RVL

This repository processes Sentinel-1 IW SLC data to estimate ocean surface radial velocity from Doppler centroid anomalies. The active code path is centered on Sentinel-1. Older BIOMASS-related utilities are still present, but they are not the main pipeline.

## Sentinel-1 Processing Model

The retrieval is built around the Sentinel-1 RVL chain:

1. Read Sentinel-1 SAFE annotation and burst SLC data.
2. Deramp the TOPS burst.
3. Estimate azimuth correlation or Doppler per block.
4. Convert Doppler to Doppler centroid anomaly by subtracting geometry and sideband terms.
5. Convert the anomaly to radial velocity.
6. Geolocate the result.
7. Apply optional metocean corrections and comparisons:
   ERA5 Stokes drift, ERA5 wave Doppler bias, OCN mispointing, and GLO12 current comparison.

There are two main processing modes:

- `scripts/sentinel_1/pipeline.py`: the main operational Sentinel-1 workflow, including merged-burst processing and corrected radial-current outputs.
- `scripts/sentinel_1/burst_pipeline.py`: a single-burst diagnostic path used to inspect burst-level behavior without the merged workflow.

## Main Scripts

### Core Sentinel-1 package

- [`scripts/sentinel_1/pipeline.py`](/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/pipeline.py:1)
  Main Sentinel-1 RVL pipeline. This is the primary entry point for burst-wise and merged-burst retrievals, current corrections, and model comparison products.

- [`scripts/sentinel_1/rvl.py`](/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/rvl.py:1)
  Core RVL algorithms: deramping, burst merging, correlation estimation, Doppler conversion, geometry subtraction, descalloping, and geolocation.

- [`scripts/sentinel_1/burst_pipeline.py`](/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/burst_pipeline.py:1)
  Single-burst Sentinel-1 RVL workflow. Useful for debugging burst-specific failures before any burst merge or mosaic step.

- [`scripts/sentinel_1/safe_io.py`](/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/safe_io.py:1)
  Sentinel-1 SAFE discovery, annotation parsing, burst reading, and calibration/noise LUT access.

- [`scripts/sentinel_1/aux_files.py`](/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/aux_files.py:1)
  Sentinel-1 auxiliary product parsers for AUX_CAL, AUX_INS, and orbit files such as POEORB.

- [`scripts/sentinel_1/ocn_product.py`](/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/ocn_product.py:1)
  Reader for Sentinel-1 Level-2 OCN SAFE products.

- [`scripts/sentinel_1/ocn_analysis.py`](/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/ocn_analysis.py:1)
  Analysis utilities built on top of OCN RVL and OWI fields, including radial-current and mispointing-related helpers.

- [`scripts/sentinel_1/metocean.py`](/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/metocean.py:1)
  Loads and interpolates ERA5, OCN, and GLO12 fields onto the SAR grid and look direction.

- [`scripts/sentinel_1/grid_merge.py`](/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/grid_merge.py:1)
  Merges burst-level outputs onto regular grids for scene-scale comparisons.

- [`scripts/sentinel_1/plots.py`](/home/oscipal/ocean_current_retrieval/scripts/sentinel_1/plots.py:1)
  Plotting helpers for merged Sentinel-1 retrievals and model comparisons.

### Diagnostics

These scripts are for investigation and validation, not the core production path.

- [`scripts/diagnostics/pipeline_diagnostics.py`](/home/oscipal/ocean_current_retrieval/scripts/diagnostics/pipeline_diagnostics.py:1)
  Intermediate RVL diagnostics such as Doppler-centroid plots, pipeline-step plots, and mispointing checks.

- [`scripts/diagnostics/current_comparison.py`](/home/oscipal/ocean_current_retrieval/scripts/diagnostics/current_comparison.py:1)
  Burst-level comparison between the SLC-derived RVL result, OCN products, ERA5 corrections, and GLO12.

- [`scripts/diagnostics/burst_fdc_offsets.py`](/home/oscipal/ocean_current_retrieval/scripts/diagnostics/burst_fdc_offsets.py:1)
  Compares burst-wise `f_dc` behavior under different deramp and PRF conventions.

- [`scripts/diagnostics/tops_scaling.py`](/home/oscipal/ocean_current_retrieval/scripts/diagnostics/tops_scaling.py:1)
  Tests alternative TOPS scaling assumptions for Doppler retrieval.

- [`scripts/diagnostics/era5_influence.py`](/home/oscipal/ocean_current_retrieval/scripts/diagnostics/era5_influence.py:1)
  Visualizes the spatial influence of ERA5-based wind and wave corrections.

- [`scripts/diagnostics/aux_cal_check.py`](/home/oscipal/ocean_current_retrieval/scripts/diagnostics/aux_cal_check.py:1)
  Quick AUX_CAL sanity check for ambiguity ratio and sideband-bias terms.

- [`scripts/diagnostics/attitude_inspection.py`](/home/oscipal/ocean_current_retrieval/scripts/diagnostics/attitude_inspection.py:1)
  One-off diagnostic for Sentinel-1 attitude quaternion conventions used in mispointing analysis.

### Shared / legacy utilities

- [`scripts/download_era5.py`](/home/oscipal/ocean_current_retrieval/scripts/download_era5.py:1)
  Downloads ERA5 wind and wave data from a JSON config file.

- [`scripts/gamma_io.py`](/home/oscipal/ocean_current_retrieval/scripts/gamma_io.py:1)
  Shared readers for GAMMA-format SLC products and related metadata. Still used by some legacy and BIOMASS-oriented code paths.

## Configuration and Inputs

- [`config/download_era5.json`](/home/oscipal/ocean_current_retrieval/config/download_era5.json:1)
  Default configuration for ERA5 downloads.

Typical Sentinel-1 processing requires:

- Sentinel-1 IW SLC SAFE
- POEORB orbit file
- AUX_CAL file
- optional OCN SAFE
- optional ERA5 wind and wave NetCDF files
- optional GLO12 ocean current NetCDF file

## Data Sources

- Sentinel-1 SLC and OCN products: ESA Copernicus
- POEORB / AUX files: ESA Sentinel-1 auxiliary products
- ERA5 wind and waves: Copernicus Climate Data Store
- GLO12 ocean currents: Copernicus Marine Service

## Setup

```bash
git clone git@github.com:oscipal/ocean_current_retrieval.git
cd ocean_current_retrieval
conda create --name ocr --file requirements.txt
conda activate ocr
```
