# Ocean Current Retrieval from SAR Doppler Centroid Anomalies

Retrieval of ocean surface current velocities from spaceborne SAR imagery using the **Doppler Centroid Anomaly (DCA)** method. The measured Doppler centroid of an SAR scene contains contributions from satellite geometry, Earth rotation, surface waves, and ocean currents. By isolating the anomalous component, radial surface current velocities can be estimated.

Currently targets ESA **BIOMASS** (P-band, ~435 MHz) and **Sentinel-1** (C-band) SAR data.

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
  io.py                   # SLC and annotation file I/O
  doppler.py              # FFT and CDE Doppler centroid estimators, SNR utilities
  geometry.py             # Geometry DC estimation from annotation polynomials
  ocean_currents.py       # Copernicus Marine NetCDF loading and GeoTIFF export
  plotting.py             # DC map and spectrum visualisation

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

---

## Usage

The `scripts/` package can be imported directly in notebooks or scripts:

```python
from scripts.io import read_slc, parse_slc_par
from scripts.doppler import fft_doppler, cde_doppler, spectral_snr, dc_precision
from scripts.geometry import estimate_geom_doppler

# Estimate Doppler centroids
dc_img, freqs, spectrum = fft_doppler(SLC_PATH, SLC_PAR_PATH, win_az=512, win_rg=100, stride_az=256, stride_rg=100)

# Subtract geometry contribution
geom_dc = estimate_geom_doppler(ANNOT_XML, doppler_img=dc_img)
dca = dc_img - geom_dc

# Check SNR
snr, noise = spectral_snr(spectrum[0, 0])
sigma_dc = dc_precision(snr, prf=1/dt_a, win_az=512, win_rg=100)
print(f"DC estimate std: {sigma_dc:.4f} Hz")
```
