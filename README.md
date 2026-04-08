# Ocean Current Retrieval from SAR Doppler Centroid Anomalies

Retrieval of ocean surface current velocities from spaceborne SAR imagery using the **Doppler Centroid Anomaly (DCA)** method. The measured Doppler centroid of an SAR scene contains contributions from satellite geometry, Earth rotation, surface waves, and ocean currents. By isolating the anomalous component, radial surface current velocities can be estimated.

Currently targets ESA **BIOMASS** (P-band, ~435 MHz) and **Sentinel-1** (C-band) SAR data.

For Sentinel-1, this repository currently contains two separate code paths:
- `scripts/s1_rvl.py` and `scripts/s1_ocn.py` work from Level-1 SLC data and implement the retrieval side of the project.
- `scripts/s1_ocn_product.py` is a standalone reader for Level-2 OCN SAFE products. It opens the delivered NetCDF files and returns the `rvl`, `owi`, and `osw` groups as `xarray` datasets. It is useful for inspection and comparison, but is not currently wired into the main analysis workflow elsewhere in this repo.

---

## Method Overview

```
Measured DC  =  Geometry DC  +  Earth rotation DC  +  Wave DC  +  Current DC
                     ↑                                               ↑
              (from annotation             DCA =  Measured DC  -  Geometry DC  -  ...
               polynomials)
```

1. Estimate the measured Doppler centroid from the SLC using FFT or CDE
2. Subtract the geometry-induced DC (from annotation polynomials)
3. Calibrate instrument bias using a desert scene (stationary surface → residual = bias)
4. Subtract remaining contributions (Earth rotation, wind/waves)
5. Convert residual DCA to radial current velocity: `v_r = DCA * λ / (2 sin θ)`

---

## Repository Structure

```
notebooks/
  desert_DCA.ipynb        # DC estimation and DCA computation from BIOMASS scenes
  DCA_predictions.ipynb   # Predicted DCA from reference ocean current data
  DCA_approximation.ipynb # Combined DC measurement and geometry correction
  s1_desert.ipynb         # Sentinel-1 burst-level DC estimation (desert calibration)

scripts/
  io.py                   # Generic SLC file I/O
  s1_io.py                # Sentinel-1 SAFE discovery and annotation parsing
  s1_rvl.py               # Sentinel-1 RVL retrieval from SLC bursts
  s1_ocn.py               # Sentinel-1 OCN-style products derived from SLC inputs
  ocean_currents.py       # Copernicus Marine NetCDF loading and GeoTIFF export
  plotting.py             # DC map and spectrum visualisation
  s1_ocn_product.py       # Direct Sentinel-1 Level-2 OCN SAFE reader

data/
  Biomass_data/           # BIOMASS SLC scenes (.slc, .slc.par, annotation XML)
  S1_data/                # Sentinel-1 burst SLC data
  current_data/           # Reference ocean current NetCDF (Copernicus Marine)
```

---

## Data Sources

| Data | Product | Resolution |
|---|---|---|
| BIOMASS L1a SLC | [ESA MAAP Explorer](https://explorer.maap.eo.esa.int/?q=BiomassLevel1a) | — |
| Ocean currents (NRT) | [MULTIOBS_GLO_PHY_MYNRT_015_003](https://data.marine.copernicus.eu/product/MULTIOBS_GLO_PHY_MYNRT_015_003/description) | 0.125° |
| Ocean currents (forecast) | [GLOBAL_ANALYSISFORECAST_PHY_001_024](https://data.marine.copernicus.eu/product/GLOBAL_ANALYSISFORECAST_PHY_001_024/description) | 0.083° |
| Ionosphere TEC | [https://cddis.nasa.gov/archive/gnss/products/ionex/2026/] | - |

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
