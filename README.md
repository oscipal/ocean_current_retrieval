# Ocean Current Retrieval from Sentinel-1 Doppler

Retrieve the **sea surface radial current** from Sentinel-1 Interferometric
Wide (IW) SLC data using the Doppler Centroid Anomaly (DCA) method. The
Doppler centroid is estimated for each TOPS burst, the geometry and
instrument biases are removed, the wind/wave (geophysical) Doppler is
subtracted, and the result is gridded into a current map and validated
against in-situ drifters.

---

## Quick start

```bash
conda create -n ocean python=3.12 && conda activate ocean
pip install -r requirements.txt
# GAMMA SAR processor must be installed separately and on PATH.

# Run the pipeline on one scene (default = custom pipeline):
python scripts/run_pipeline.py

# Run the hybrid configuration instead:
python scripts/run_pipeline.py --hybrid
```

`scripts/run_pipeline.py` writes a NetCDF current map and a quicklook PNG to
`data/current/`, and prints summary statistics.

```
python scripts/run_pipeline.py [--hybrid] [--scene scene1] [--subswath iw1]
                       [--pol vv] [--data-root data] [--out-dir data/current]
```

---

## The two pipelines

| | Custom pipeline (default) | Hybrid (`--hybrid`) |
|---|---|---|
| Doppler | estimated per burst with GAMMA (lag-1) | observed Doppler from the Sentinel-1 OCN product |
| Geometry | removed from the precise orbit | from the OCN product |
| Corrections | sideband, descalloping, mispointing, Stokes, wave (Mouche) | sideband, mispointing, Stokes, wave (Mouche) |
| Works on | radar range/azimuth grid, then gridded | OCN product grid |

Both end in `v_current_ocn`, the radial surface current including the
ocean-product mispointing term.

---

## Repository layout

```
scripts/run_pipeline.py    Entry point: default custom pipeline, --hybrid for the hybrid
requirements.txt           Python dependencies (env "ocean")
scripts/
  sentinel_1/              Core library (pipeline, RVL, gamma_variants, grid_merge,
                           metocean, cdop, ocn_product, safe_io, ...)
  download/                Data download (ERA5, matched Sentinel-1 + drifter scenes)
  diagnostics/             Doppler/deramp diagnostics (spectrogram, unwrapped grid, ...)
  figures/                 Figure generation (report figures, comparison maps, scatter)
  validation/              Drifter validation and method sweeps
notebooks/
  1_pipeline_walkthrough.ipynb    Custom pipeline, step by step
  2_hybrid_pipeline.ipynb         Hybrid pipeline
  3_evaluation_comparison.ipynb   Drifter validation + model comparison
data/                      Sentinel-1 SAFE, ERA5, GLO12, drifters, outputs
notes/                     LaTeX project report (notes/files/) and its figures
config/                    Download configuration (e.g. download_era5.json)
_archive/                  Old notebooks, BIOMASS utilities, scratch (not maintained)
```

The core library is imported as `scripts.sentinel_1.*` from the repo root,
so run scripts and notebooks from the repository root.

---

## Notebooks

- **`1_pipeline_walkthrough.ipynb`** — runs the custom pipeline and shows the
  field after every step (Doppler domain in Hz, then velocity in m/s).
- **`2_hybrid_pipeline.ipynb`** — runs the hybrid configuration and shows the
  ocean-product Doppler and the resulting current.
- **`3_evaluation_comparison.ipynb`** — validates against drifters and
  compares the custom pipeline, the hybrid, the operational OCN product and
  the GLO12 model; shows the scatter and the cumulative effect of each
  correction. Uses the precomputed tables in `data/drifters/`.

---

## Validation and figures

```bash
# Drifter validation across scenes (writes data/drifters/validation_results.csv):
python scripts/validation/run_drifter_validation_all.py --help

# Regenerate the report figures into notes/files/figures/:
python scripts/figures/make_report_figures.py

# Four-way comparison map (custom | hybrid | OCN | GLO12):
python scripts/figures/make_poster_figure.py --current
```

---

## Data

Sentinel-1 IW SLC + Level-2 OCN scenes go under `data/sentinel-1/<scene>/`
(`S1A_IW_SLC.SAFE`, `S1A_IW_OCN.SAFE`, the `POEORB` orbit `.EOF`, and the
`AUX_CAL` calibration). ERA5 wind/wave and the GLO12 current go under
`data/era5_data/<scene>/`. Use the download scripts in `scripts/download/`
to fetch ERA5 (`cdsapi`) and the matched CMEMS GLO12/drifter data
(`copernicusmarine`).
