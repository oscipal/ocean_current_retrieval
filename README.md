# Ocean Current Retrieval from SAR Doppler Centroid Anomalies

Retrieval of ocean surface current radial velocities from spaceborne SAR imagery using the Doppler Centroid Anomaly (DCA) method. The measured Doppler centroid of a SAR scene contains contributions from satellite geometry, instrument biases, surface waves and ocean currents. By subtracting all known contributions the residual gives the radial surface current velocity.

The main focus is Sentinel-1 IW TOPS data in C-band. Some earlier exploratory work on BIOMASS P-band data is also in the repo.

---

## Method Overview

```
Observed DC  =  Geometry DC  +  Instrument bias  +  Wave DC  +  Mispointing  +  Current DC
```

1. Read the SLC burst and deramp the TOPS chirp
2. Estimate lag-0/lag-1 azimuth correlation coefficients (p0, p1) per block
3. Convert p1 to Doppler centroid; apply AUX_CAL ambiguity and sideband corrections
4. Subtract geometry Doppler computed from POEORB precision orbit
5. Subtract ERA5 Stokes drift and wave Doppler bias
6. Apply OCN rvlDcMiss mispointing correction
7. Convert residual DCA to radial velocity: v_r = DCA * lambda / 2

The full pipeline with plots and explanations is in `notebooks/rvl_pipeline_walkthrough.ipynb`.

---

## Repository Structure

```
notebooks/
  rvl_pipeline_walkthrough.ipynb  # Main step-by-step Sentinel-1 RVL pipeline
  s1_ocean.ipynb                  # Ocean scene processing
  s1_ocn_burst.ipynb              # Burst-level OCN product exploration
  s1_ocn_gamma.ipynb              # Ambiguity ratio investigation
  s1_ocn_safe.ipynb               # OCN SAFE product reader/inspection
  desert_DCA.ipynb                # DC estimation from BIOMASS desert scenes
  desert_DCA_new.ipynb            # Updated desert DCA workflow
  DCA_predictions.ipynb           # Predicted DCA from reference current data
  DCA_approximation.ipynb         # DC measurement and geometry correction
  s1_desert.ipynb                 # Sentinel-1 burst-level DC estimation
  nisar.ipynb                     # NISAR exploratory work

scripts/
  s1_io.py                        # Sentinel-1 SAFE discovery and annotation parsing
  s1_rvl.py                       # Core RVL retrieval (deramp, correlation, Doppler)
  s1_rvl_burst.py                 # Burst-level RVL helpers
  s1_aux.py                       # AUX_CAL and POEORB parsing and application
  s1_ocn.py                       # OCN-style products derived from SLC inputs
  s1_ocn_product.py               # Level-2 OCN SAFE reader (returns xarray datasets)
  rvl_current.py                  # Ocean current extraction (Stokes, wave bias, GLO12)
  compare_to_ocn.py               # Comparison between SLC-derived and OCN RVL
  download_era5_scene.py          # ERA5 wind and wave data download for a scene
  doppler.py                      # Generic Doppler estimation utilities
  ocean_currents.py               # Copernicus Marine NetCDF loading and GeoTIFF export
  io.py                           # Generic SLC file I/O
  plotting.py                     # Visualisation helpers
  corrections/
    geometry.py                   # Geometry Doppler and POEORB correction
    bias.py                       # Instrument bias corrections
    ionosphere.py                 # Ionospheric correction utilities

plots/                            # Output figures
papers/                           # Reference algorithm documents and papers
shell_commands/
  burst_extraction.sh             # Burst extraction helper
```

---

## Data Sources

| Data | Product | Resolution |
|------|---------|------------|
| Sentinel-1 SLC | ESA Copernicus Open Access Hub | IW ~20 x 5 m |
| Sentinel-1 OCN | ESA Copernicus Open Access Hub | ~1 km |
| AUX_CAL / POEORB | ESA auxiliary file server | |
| Ocean currents | [CMEMS GLO12 GLOBAL_ANALYSISFORECAST_PHY_001_024](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description) | 1/12° |
| ERA5 wind and waves | [Copernicus Climate Data Store](https://cds.climate.copernicus.eu) | ~31 km |
| BIOMASS L1a SLC | [ESA MAAP Explorer](https://explorer.maap.eo.esa.int/?q=BiomassLevel1a) | |

---

## Setup

Clone the repository:

```bash
git clone git@github.com:oscipal/ocean_current_retrieval.git
cd ocean_current_retrieval/
```

Create a conda environment with all dependencies:

```bash
conda create --name ocr --file requirements.txt
conda activate ocr
```
