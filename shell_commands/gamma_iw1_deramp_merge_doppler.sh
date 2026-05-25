#!/bin/bash
# GAMMA-based pipeline for Sentinel-1 IW1:
#   1. Import SAFE → GAMMA SLC
#   2. Apply POEORB precise orbit
#   3. Deramp (SLC_deramp_ScanSAR, mode=0, ESA TOPS deramping formula)
#   4. Mosaic deramped bursts into a continuous SLC (SLC_mosaic_ScanSAR)
#   5. Estimate Doppler centroid on the mosaic (doppler_2d_SLC)
#
# Run from the repo root:  bash shell_commands/gamma_iw1_deramp_merge_doppler.sh

set -euo pipefail

# ── Input paths ───────────────────────────────────────────────────────────────
SAFE=data/sentinel-1/scene1/S1A_IW_SLC.SAFE
POEORB=data/sentinel-1/S1A_OPER_AUX_POEORB_OPOD_20260225T070420_V20260204T225942_20260206T005942.EOF

# IW1 VV files inside the SAFE
MEAS_DIR=${SAFE}/measurement
ANNOT_DIR=${SAFE}/annotation
CAL_DIR=${ANNOT_DIR}/calibration

TIFF=$(ls   ${MEAS_DIR}/s1a-iw1-slc-vv-*.tiff | head -1)
ANNOT=$(ls  ${ANNOT_DIR}/s1a-iw1-slc-vv-*.xml | head -1)
CAL=$(ls    ${CAL_DIR}/calibration-s1a-iw1-slc-vv-*.xml | head -1)
NOISE=$(ls  ${CAL_DIR}/noise-s1a-iw1-slc-vv-*.xml | head -1)

# ── Doppler estimation parameters ────────────────────────────────────────────
BLSZ=256           # block size in azimuth lines passed to doppler_2d_SLC

# ── Output directory ──────────────────────────────────────────────────────────
OUTDIR=data/sentinel-1/gamma_iw1
mkdir -p ${OUTDIR}

ID=20260205_iw1_vv          # base name used for all output files

SLC=${OUTDIR}/${ID}.slc
SLC_PAR=${OUTDIR}/${ID}.slc.par
TOPS_PAR=${OUTDIR}/${ID}.slc.tops_par

SLC_DR=${OUTDIR}/${ID}.deramp.slc
SLC_DR_PAR=${OUTDIR}/${ID}.deramp.slc.par
TOPS_DR_PAR=${OUTDIR}/${ID}.deramp.slc.tops_par

MOSAIC=${OUTDIR}/${ID}.deramp.mosaic.slc
MOSAIC_PAR=${OUTDIR}/${ID}.deramp.mosaic.slc.par

DOP2D=${OUTDIR}/${ID}.dop2d.txt

# SLC_tab files (3-column: SLC  SLC_par  TOPS_par)
SLC_TAB=${OUTDIR}/${ID}.SLC_tab
SLC_DR_TAB=${OUTDIR}/${ID}.deramp.SLC_tab

echo "================================================================"
echo "Step 1: Import Sentinel-1 IW1 VV from SAFE"
echo "================================================================"
par_S1_SLC \
    ${TIFF} \
    ${ANNOT} \
    ${CAL} \
    ${NOISE} \
    ${SLC_PAR} \
    ${SLC} \
    ${TOPS_PAR}

echo ""
echo "================================================================"
echo "Step 2: Replace predicted orbit with POEORB state vectors"
echo "================================================================"
S1_OPOD_vec ${SLC_PAR} ${POEORB}

echo ""
echo "================================================================"
echo "Step 3: Deramp – remove TOPS azimuth phase ramp"
echo "        (SLC_deramp_ScanSAR mode=0, phflg=1 saves .dph for inspection)"
echo "================================================================"

# Build the input SLC_tab (one line for IW1)
printf "%s\t%s\t%s\n" ${SLC} ${SLC_PAR} ${TOPS_PAR} > ${SLC_TAB}

# Build the output (deramped) SLC_tab
printf "%s\t%s\t%s\n" ${SLC_DR} ${SLC_DR_PAR} ${TOPS_DR_PAR} > ${SLC_DR_TAB}

# mode 0 = subtract Doppler phase (deramp); phflg 1 = save .dph deramp phase
SLC_deramp_ScanSAR ${SLC_TAB} ${SLC_DR_TAB} 0 1

echo ""
echo "================================================================"
echo "Step 4: Mosaic deramped bursts into a continuous SLC"
echo "        rlks=1 azlks=1 for full-resolution SLC"
echo "================================================================"
SLC_mosaic_ScanSAR ${SLC_DR_TAB} ${MOSAIC} ${MOSAIC_PAR} 1 1 1

echo ""
echo "================================================================"
echo "Step 5: Estimate Doppler centroid on the mosaic"
echo "        Block size ${BLSZ} lines; b0 (along-track trend) enabled"
echo "================================================================"
# doppler_2d_SLC <SLC> <SLC_par> <dop2d> [loff] [blsz] [nbl] [a2] [b0] [b1] [c0] [namb]
doppler_2d_SLC ${MOSAIC} ${MOSAIC_PAR} ${DOP2D} 0 ${BLSZ} - 0 1 0 0

echo ""
echo "================================================================"
echo "Step 6: Rasterise dop2d text → 2-D numpy arrays"
echo "        fd_measured, fd_model, fd_diff  (n_blocks × n_range_cols)"
echo "================================================================"

DOP_NPY=${OUTDIR}/${ID}.dop2d.npz

python3 - "${DOP2D}" "${DOP_NPY}" "${MOSAIC_PAR}" <<'PYEOF'
import sys
import re
import numpy as np

dop2d_path, out_npy, par_path = sys.argv[1], sys.argv[2], sys.argv[3]

# ── Read the 4-column text file ───────────────────────────────────────────────
# doppler_2d_SLC writes:  range_sample  fd_measured  fd_model  fd_diff
# The same range-sample sequence repeats once per azimuth block.
data = np.loadtxt(dop2d_path)          # shape (N_blocks * N_range_cols, 4)
range_col = data[:, 0]

# Detect where the range index resets (start of a new block)
resets = np.where(np.diff(range_col) < 0)[0] + 1
n_range = int(resets[0]) if len(resets) > 0 else len(data)
n_blocks = len(data) // n_range
# Truncate any incomplete trailing block before reshaping
data = data[:n_blocks * n_range]

fd_meas  = data[:, 1].reshape(n_blocks, n_range).astype(np.float32)
fd_model = data[:, 2].reshape(n_blocks, n_range).astype(np.float32)
fd_diff  = data[:, 3].reshape(n_blocks, n_range).astype(np.float32)

# ── Read SLC par for image dimensions and block geometry ──────────────────────
def read_par(path, key):
    with open(path) as f:
        for line in f:
            if line.startswith(key + ':'):
                return line.split(':')[1].split()[0]
    return None

n_az   = int(read_par(par_path, 'azimuth_lines'))
n_rg   = int(read_par(par_path, 'range_samples'))
prf    = float(read_par(par_path, 'prf'))
blsz   = n_az // n_blocks          # approximate lines per block

# Range axis in metres (near-range + sample * spacing)
r0     = float(read_par(par_path, 'near_range_slc'))
dr     = float(read_par(par_path, 'range_pixel_spacing'))
# range_col holds the sample indices output by doppler_2d_SLC
range_m = r0 + data[:n_range, 0] * dr

# Azimuth block-centre times (seconds from image start)
ati     = float(read_par(par_path, 'azimuth_line_time'))
az_time = (np.arange(n_blocks) + 0.5) * blsz * ati

# ── Save ──────────────────────────────────────────────────────────────────────
np.savez(
    out_npy,
    fd_measured = fd_meas,    # Hz  (n_blocks × n_range_cols)
    fd_model    = fd_model,   # Hz
    fd_diff     = fd_diff,    # Hz  = measured − model
    range_m     = range_m,    # slant range [m]  (n_range_cols,)
    az_time_s   = az_time,    # azimuth centre time of each block [s]  (n_blocks,)
    n_az_lines  = np.array(n_az),
    n_rg_samples= np.array(n_rg),
    prf_hz      = np.array(prf),
    blsz_lines  = np.array(blsz),
)
print(f"  Saved {n_blocks} blocks × {n_range} range cols  →  {out_npy}")
print(f"  fd_measured  shape: {fd_meas.shape}   dtype: {fd_meas.dtype}")
print(f"  range_m      [{range_m[0]:.0f} … {range_m[-1]:.0f}] m")
print(f"  az_time_s    [{az_time[0]:.3f} … {az_time[-1]:.3f}] s")
PYEOF

echo ""
echo "================================================================"
echo "Done.  Key output files:"
echo "  Deramped burst SLC : ${SLC_DR}"
echo "  Mosaic SLC         : ${MOSAIC}"
echo "  Mosaic SLC par     : ${MOSAIC_PAR}"
echo "  Doppler text       : ${DOP2D}"
echo "  Doppler arrays     : ${DOP_NPY}"
echo ""
echo "Quick look in Python:"
echo "  import numpy as np, matplotlib.pyplot as plt"
echo "  d = np.load('${DOP_NPY}')"
echo "  plt.figure(); plt.imshow(d['fd_measured'], aspect='auto',"
echo "      extent=[d['range_m'][0], d['range_m'][-1],"
echo "              d['az_time_s'][-1], d['az_time_s'][0]])"
echo "  plt.colorbar(label='Doppler centroid (Hz)')"
echo "  plt.xlabel('Slant range (m)'); plt.ylabel('Azimuth time (s)')"
echo "  plt.title('IW1 VV – measured Doppler centroid (deramped mosaic)')"
echo "  plt.show()"
echo "================================================================"
