"""Drive GAMMA ``doppler_2d_SLC`` for mosaic-first / mosaic-last block-size variants.

The repo's GAMMA pipeline (``shell_commands/gamma_iw1_deramp_merge_doppler.sh``)
runs ``deramp -> SLC_mosaic_ScanSAR -> doppler_2d_SLC`` — i.e. it *mosaics first*
and estimates the Doppler centroid once on the continuous mosaic SLC.

This module adds the two variants the comparison notebook needs:

* :func:`gamma_doppler_mosaic_first` — re-run ``doppler_2d_SLC`` on the existing
  deramped mosaic SLC with an arbitrary block size.
* :func:`gamma_doppler_mosaic_last` — estimate the Doppler centroid *per
  deramped burst* (no mosaic before estimation), then stitch the per-burst
  Doppler grids in azimuth.  GAMMA's ``SLC_mosaic_ScanSAR`` only mosaics complex
  SLC data, not Doppler grids, so the final stitch is a plain azimuth
  concatenation — the "do it with our pipeline" fallback.

Both write a ``.npz`` with the same schema as the shell script
(``fd_measured, fd_model, fd_diff, range_m, az_time_s``) so the result can be
fed straight into :func:`pipeline.run_gamma_dop2d_pipeline`.

Note: ``doppler_2d_SLC`` enforces a minimum block size of 256 lines.
"""

from __future__ import annotations

import contextlib
import glob
import os
import shutil
import subprocess
import tempfile
from datetime import datetime
from typing import Iterator

import numpy as np

# Block sizes below this are rejected by doppler_2d_SLC.
GAMMA_MIN_BLSZ = 256

# Default GAMMA install location, used if the binary is not on PATH.
_GAMMA_FALLBACK_BIN = "/opt/GAMMA_SOFTWARE-20251203/ISP/bin"


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gamma_bin(name: str) -> str:
    """Locate a GAMMA executable on PATH, falling back to the known install dir."""
    exe = shutil.which(name)
    if exe:
        return exe
    fallback = os.path.join(_GAMMA_FALLBACK_BIN, name)
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError(
        f"GAMMA program {name!r} not found on PATH or in {_GAMMA_FALLBACK_BIN}"
    )


def _read_par(path: str, key: str) -> str | None:
    """Return the first whitespace token after ``key:`` in a GAMMA par file."""
    with open(path) as f:
        for line in f:
            if line.startswith(key + ":"):
                return line.split(":", 1)[1].split()[0]
    return None


def _parse_tops_par(tops_par: str) -> tuple[int, int, list[dict]]:
    """Parse a GAMMA ``.tops_par`` for burst geometry.

    Returns
    -------
    n_bursts : int
    lines_per_burst : int
    bursts : list of dict
        One dict per burst with ``start_time`` [s], ``first_valid_line`` and
        ``last_valid_line`` (burst-relative line indices).
    """
    fields: dict[str, list[str]] = {}
    with open(tops_par) as f:
        for line in f:
            if ":" in line:
                key, _, val = line.partition(":")
                fields[key.strip()] = val.split()

    n_bursts = int(fields["number_of_bursts"][0])
    lpb = int(fields["lines_per_burst"][0])
    bursts = []
    for i in range(1, n_bursts + 1):
        rec = {
            "start_time": float(fields[f"burst_start_time_{i}"][0]),
            "first_valid_line": int(fields[f"first_valid_line_{i}"][0]),
            "last_valid_line": int(fields[f"last_valid_line_{i}"][0]),
        }
        # Per-burst Doppler polynomial that SLC_deramp_ScanSAR demodulated by.
        # Stored as ``doppler_polynomial_N`` (coefficients in slant-range time
        # relative to ``doppler_srdelay_N``) in the .tops_par.
        if f"doppler_polynomial_{i}" in fields:
            rec["doppler_polynomial"] = [
                float(x) for x in fields[f"doppler_polynomial_{i}"]
            ]
            rec["doppler_srdelay"] = float(fields[f"doppler_srdelay_{i}"][0])
        bursts.append(rec)
    return n_bursts, lpb, bursts


def _run_doppler_2d(
    slc: str,
    slc_par: str,
    dop2d_txt: str,
    loff: int,
    blsz: int,
    nbl: int | None,
) -> None:
    """Invoke ``doppler_2d_SLC`` (a2=0, b0=1, b1=0, c0=0 — as in the shell script)."""
    if blsz < GAMMA_MIN_BLSZ:
        raise ValueError(
            f"doppler_2d_SLC requires blsz >= {GAMMA_MIN_BLSZ}, got {blsz}"
        )
    cmd = [
        _gamma_bin("doppler_2d_SLC"),
        slc, slc_par, dop2d_txt,
        str(loff), str(blsz), str(nbl) if nbl else "-",
        "0", "1", "0", "0",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"doppler_2d_SLC failed (exit {proc.returncode})\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout:\n{proc.stdout}\n  stderr:\n{proc.stderr}"
        )


def _burst_window(lpb: int, overlap_lines: int) -> np.ndarray:
    """Raised-cosine partition-of-unity window for one burst.

    Returns a length-``lpb`` window ``w`` where the first / last
    ``overlap_lines`` are sin/cos tapers (so ``w_i(end)^2 + w_{i+1}(start)^2 = 1``
    for adjacent bursts) and the core is constant 1.  Mirrors the burst window
    used by :func:`rvl._blend_burst_profiles`.
    """
    w = np.ones(lpb, dtype=np.float64)
    if overlap_lines <= 0:
        return w
    k = np.arange(overlap_lines)
    ramp = np.sin((k + 0.5) / overlap_lines * np.pi / 2.0)   # 0 -> 1
    w[:overlap_lines] = ramp
    w[-overlap_lines:] = ramp[::-1]                          # 1 -> 0
    return w


def _resample_range(arr: np.ndarray, src_rg: np.ndarray, dst_rg: np.ndarray) -> np.ndarray:
    """Linearly resample a ``(n_blocks, n_range)`` grid onto ``dst_rg`` range samples."""
    if src_rg.shape == dst_rg.shape and np.allclose(src_rg, dst_rg):
        return arr
    return np.stack(
        [np.interp(dst_rg, src_rg, row) for row in arr], axis=0
    ).astype(np.float32)


def _parse_dop2d(txt_path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a ``doppler_2d_SLC`` text file into ``(fd_meas, fd_model, fd_diff, rg_idx)``.

    The text file holds four columns — ``range_sample fd_measured fd_model
    fd_diff`` — with the range-sample sequence repeating once per azimuth block.
    """
    data = np.loadtxt(txt_path)
    range_col = data[:, 0]
    resets = np.where(np.diff(range_col) < 0)[0] + 1
    n_range = int(resets[0]) if len(resets) > 0 else len(data)
    n_blocks = len(data) // n_range
    data = data[: n_blocks * n_range]

    fd_meas = data[:, 1].reshape(n_blocks, n_range).astype(np.float32)
    fd_model = data[:, 2].reshape(n_blocks, n_range).astype(np.float32)
    fd_diff = data[:, 3].reshape(n_blocks, n_range).astype(np.float32)
    rg_idx = data[:n_range, 0]
    return fd_meas, fd_model, fd_diff, rg_idx


# ─────────────────────────────────────────────────────────────────────────────
# Variant 1 — mosaic first, then estimate Doppler on the mosaic
# ─────────────────────────────────────────────────────────────────────────────

def _compute_doppler_mosaic_first(
    blsz: int,
    mosaic_slc: str,
    mosaic_par: str,
    txt_path: str,
) -> dict:
    """Run ``doppler_2d_SLC`` once on a mosaic SLC and return the result in memory."""
    print(f"  doppler_2d_SLC on mosaic, blsz={blsz} ...")
    _run_doppler_2d(mosaic_slc, mosaic_par, txt_path, loff=0, blsz=blsz, nbl=None)

    fd_meas, fd_model, fd_diff, rg_idx = _parse_dop2d(txt_path)
    n_blocks = fd_meas.shape[0]
    r0 = float(_read_par(mosaic_par, "near_range_slc"))
    dr = float(_read_par(mosaic_par, "range_pixel_spacing"))
    ati = float(_read_par(mosaic_par, "azimuth_line_time"))
    range_m = r0 + rg_idx * dr
    az_time = (np.arange(n_blocks) + 0.5) * blsz * ati
    print(f"  computed {n_blocks} blocks x {fd_meas.shape[1]} range cols")
    return {
        "fd_measured": fd_meas,
        "fd_model": fd_model,
        "fd_diff": fd_diff,
        "range_m": range_m,
        "az_time_s": az_time,
        "blsz_lines": int(blsz),
        "mosaic_mode": "mosaic_first",
    }


def gamma_doppler_mosaic_first(
    blsz: int,
    gamma_dir: str = "data/sentinel-1/gamma_iw1",
    base_id: str = "20260205_iw1_vv",
    out_dir: str | None = None,
    overwrite: bool = False,
    return_dict: bool = False,
) -> "str | dict":
    """Estimate the Doppler centroid on the pre-built deramped mosaic SLC.

    This mirrors the shell script: the bursts are already deramped and mosaicked
    into ``{base_id}.deramp.mosaic.slc``; only ``doppler_2d_SLC`` is re-run, here
    with an arbitrary ``blsz``.

    Parameters
    ----------
    blsz : int
        ``doppler_2d_SLC`` azimuth block size in lines (>= 256).
    gamma_dir : str
        Directory holding the GAMMA products from the shell script.
    base_id : str
        Output basename used by the shell script.
    out_dir : str or None
        Where to write the ``.npz`` / ``.txt`` (defaults to ``gamma_dir``).
    overwrite : bool
        Re-run even when a cached ``.npz`` already exists.
    return_dict : bool
        If True, skip the ``.npz`` save and return the result dict in memory
        (per-call ``.txt`` written to a tempdir).

    Returns
    -------
    str or dict
        Path to the ``.npz`` (default), or the result dict if
        ``return_dict=True``.
    """
    mosaic = os.path.join(gamma_dir, f"{base_id}.deramp.mosaic.slc")
    mosaic_par = mosaic + ".par"

    if not os.path.exists(mosaic):
        raise FileNotFoundError(
            f"Deramped mosaic SLC not found: {mosaic}\n"
            f"  Run shell_commands/gamma_iw1_deramp_merge_doppler.sh first,"
            f" or use gamma_prep_scene(..., build_mosaic=True)."
        )

    if return_dict:
        with tempfile.TemporaryDirectory(prefix="gamma_dop2d_") as td:
            txt = os.path.join(td, f"{base_id}.dop2d.mosaic_first.blsz{blsz}.txt")
            return _compute_doppler_mosaic_first(blsz, mosaic, mosaic_par, txt)

    out_dir = out_dir or gamma_dir
    npz = os.path.join(out_dir, f"{base_id}.dop2d.mosaic_first.blsz{blsz}.npz")
    if os.path.exists(npz) and not overwrite:
        print(f"  [cached] {npz}")
        return npz

    txt = npz[:-4] + ".txt"
    result = _compute_doppler_mosaic_first(blsz, mosaic, mosaic_par, txt)

    np.savez(
        npz,
        fd_measured=result["fd_measured"],
        fd_model=result["fd_model"],
        fd_diff=result["fd_diff"],
        range_m=result["range_m"],
        az_time_s=result["az_time_s"],
        blsz_lines=np.array(blsz),
        mosaic_mode=np.array("mosaic_first"),
    )
    print(f"  saved -> {npz}")
    return npz


# ─────────────────────────────────────────────────────────────────────────────
# In-memory scene prep — build the deramped (and optionally mosaicked) GAMMA
# products from a SAFE into a tempdir, so callers don't need to run the bash
# script up-front per scene.
# ─────────────────────────────────────────────────────────────────────────────

def _safe_subswath_files(safe_dir: str, subswath: str, pol: str) -> dict:
    """Locate the IW{1,2,3} VV/VH measurement, annotation, and cal/noise XMLs."""
    sw, pol = subswath.lower(), pol.lower()

    def _one(pattern: str) -> str:
        hits = sorted(glob.glob(pattern))
        if not hits:
            raise FileNotFoundError(f"No file matching {pattern}")
        return hits[0]

    return {
        "tiff":  _one(os.path.join(safe_dir, "measurement", f"s1?-{sw}-slc-{pol}-*.tiff")),
        "annot": _one(os.path.join(safe_dir, "annotation",  f"s1?-{sw}-slc-{pol}-*.xml")),
        "cal":   _one(os.path.join(safe_dir, "annotation", "calibration",
                                   f"calibration-s1?-{sw}-slc-{pol}-*.xml")),
        "noise": _one(os.path.join(safe_dir, "annotation", "calibration",
                                   f"noise-s1?-{sw}-slc-{pol}-*.xml")),
    }


def _safe_overpass_date(safe_dir: str) -> str:
    """Extract YYYYMMDD from the SAFE basename (the burst start date)."""
    name = os.path.basename(os.path.normpath(safe_dir))
    # S1A_IW_SLC__1SDV_20260205T... — take the first YYYYMMDDTHHMMSS token.
    for tok in name.split("_"):
        if len(tok) >= 15 and tok[8] == "T" and tok[:8].isdigit():
            return tok[:8]
    return datetime.utcnow().strftime("%Y%m%d")


def _run_gamma(cmd: list, label: str) -> None:
    """Run a GAMMA binary, echoing the command and surfacing stderr on failure."""
    print(f"  [{label}] {' '.join(cmd)}")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{label} failed (exit {proc.returncode})\n"
            f"  cmd: {' '.join(cmd)}\n"
            f"  stdout:\n{proc.stdout}\n  stderr:\n{proc.stderr}"
        )


@contextlib.contextmanager
def gamma_prep_scene(
    slc_safe: str,
    subswath: str,
    poeorb_path: "str | None" = None,
    polarisation: str = "vv",
    out_dir: "str | None" = None,
    base_id: "str | None" = None,
    build_mosaic: bool = False,
) -> Iterator[dict]:
    """Build GAMMA deramped products from a Sentinel-1 SAFE.

    Runs ``par_S1_SLC`` (+ optional ``S1_OPOD_vec`` if ``poeorb_path``) then
    ``SLC_deramp_ScanSAR`` for the given subswath/polarisation.  Optionally also
    runs ``SLC_mosaic_ScanSAR`` (needed for the mosaic-first variant).

    Parameters
    ----------
    out_dir : str or None
        Where to put the products.  If None (the default), a
        :class:`tempfile.TemporaryDirectory` is used and *all GAMMA outputs are
        deleted on context exit* — pass an explicit ``out_dir`` if you want to
        keep them.
    base_id : str or None
        Basename for all GAMMA outputs.  Defaults to
        ``f"{YYYYMMDD}_{subswath}_{pol}"`` (matches the shell script's
        convention).
    build_mosaic : bool
        Also run ``SLC_mosaic_ScanSAR`` to produce the mosaicked SLC.

    Yields
    ------
    dict
        Paths and identifiers for the GAMMA products:
        ``gamma_dir, base_id, slc, slc_par, tops_par, deramp_slc, deramp_par,
        deramp_tops_par`` (always); ``mosaic_slc, mosaic_par`` (if
        ``build_mosaic``).
    """
    files = _safe_subswath_files(slc_safe, subswath, polarisation)
    sw, pol = subswath.lower(), polarisation.lower()
    bid = base_id or f"{_safe_overpass_date(slc_safe)}_{sw}_{pol}"

    if out_dir is None:
        ctx = tempfile.TemporaryDirectory(prefix=f"gamma_prep_{sw}_{pol}_")
    else:
        os.makedirs(out_dir, exist_ok=True)
        ctx = contextlib.nullcontext(out_dir)

    with ctx as gamma_dir:
        slc        = os.path.join(gamma_dir, f"{bid}.slc")
        slc_par    = slc + ".par"
        tops_par   = slc + ".tops_par"
        slc_dr     = os.path.join(gamma_dir, f"{bid}.deramp.slc")
        slc_dr_par = slc_dr + ".par"
        tops_dr_pr = slc_dr + ".tops_par"
        slc_tab    = os.path.join(gamma_dir, f"{bid}.SLC_tab")
        slc_dr_tab = os.path.join(gamma_dir, f"{bid}.deramp.SLC_tab")

        # Step 1 — import SAFE → GAMMA SLC.
        _run_gamma(
            [_gamma_bin("par_S1_SLC"),
             files["tiff"], files["annot"], files["cal"], files["noise"],
             slc_par, slc, tops_par],
            "par_S1_SLC",
        )

        # Step 2 — replace predicted with precise orbit (POEORB), if provided.
        if poeorb_path is not None:
            _run_gamma(
                [_gamma_bin("S1_OPOD_vec"), slc_par, str(poeorb_path)],
                "S1_OPOD_vec",
            )

        # Step 3 — deramp (TOPS phase removed, mode=0).
        with open(slc_tab, "w") as f:
            f.write(f"{slc}\t{slc_par}\t{tops_par}\n")
        with open(slc_dr_tab, "w") as f:
            f.write(f"{slc_dr}\t{slc_dr_par}\t{tops_dr_pr}\n")
        _run_gamma(
            [_gamma_bin("SLC_deramp_ScanSAR"), slc_tab, slc_dr_tab, "0", "0"],
            "SLC_deramp_ScanSAR",
        )

        result = {
            "gamma_dir":       gamma_dir,
            "base_id":         bid,
            "slc":             slc,
            "slc_par":         slc_par,
            "tops_par":        tops_par,
            "deramp_slc":      slc_dr,
            "deramp_par":      slc_dr_par,
            "deramp_tops_par": tops_dr_pr,
        }

        if build_mosaic:
            mosaic     = os.path.join(gamma_dir, f"{bid}.deramp.mosaic.slc")
            mosaic_par = mosaic + ".par"
            _run_gamma(
                [_gamma_bin("SLC_mosaic_ScanSAR"),
                 slc_dr_tab, mosaic, mosaic_par, "1", "1", "1"],
                "SLC_mosaic_ScanSAR",
            )
            result["mosaic_slc"] = mosaic
            result["mosaic_par"] = mosaic_par

        yield result


# ─────────────────────────────────────────────────────────────────────────────
# Variant 2 — estimate Doppler per deramped burst, mosaic (stitch) last
# ─────────────────────────────────────────────────────────────────────────────

def _normalise_demod_back(add_demod_back: "str | bool") -> str:
    if add_demod_back is True:
        return "blend"
    if add_demod_back is False:
        return "none"
    if add_demod_back not in {"none", "step", "blend", "hanning"}:
        raise ValueError(
            f"add_demod_back must be one of 'none','step','blend','hanning' "
            f"(or bool); got {add_demod_back!r}"
        )
    return add_demod_back


def _compute_doppler_mosaic_last(
    blsz: int,
    deramp_slc: str,
    deramp_par: str,
    tops_par: str,
    add_demod_back: str,
    txt_dir: str,
    txt_prefix: str = "dop2d",
) -> dict:
    """Run ``doppler_2d_SLC`` per burst and return the stitched result in memory.

    Per-burst ``.txt`` files are written into ``txt_dir`` (pass a tempdir to
    keep them ephemeral).  Returns a dict with the same schema as the ``.npz``
    written by :func:`gamma_doppler_mosaic_last`.
    """
    n_bursts, lpb, bursts = _parse_tops_par(tops_par)
    r0 = float(_read_par(deramp_par, "near_range_slc"))
    dr = float(_read_par(deramp_par, "range_pixel_spacing"))
    ati = float(_read_par(deramp_par, "azimuth_line_time"))
    t0_scene = bursts[0]["start_time"]

    per_burst: list[dict] = []
    for i, b in enumerate(bursts):
        fvl, lvl = b["first_valid_line"], b["last_valid_line"]
        nbl = (lvl - fvl) // blsz
        if nbl < 1:
            print(f"  burst {i + 1}: valid lines {lvl - fvl} < blsz {blsz}, skipped")
            continue

        loff = i * lpb + fvl
        txt = os.path.join(txt_dir, f"{txt_prefix}.burst{i + 1}.txt")
        print(f"  doppler_2d_SLC burst {i + 1}/{n_bursts}, loff={loff}, nbl={nbl} ...")
        _run_doppler_2d(deramp_slc, deramp_par, txt, loff=loff, blsz=blsz, nbl=nbl)

        fd_meas, fd_model, fd_diff, rg_idx = _parse_dop2d(txt)
        n_blk = fd_meas.shape[0]

        if add_demod_back == "step" and "doppler_polynomial" in b:
            range_m_b = r0 + rg_idx * dr
            tau_b = 2.0 * range_m_b / 299_792_458.0
            dt = tau_b - b["doppler_srdelay"]
            poly_N = sum(c * dt ** k for k, c in enumerate(b["doppler_polynomial"]))
            poly_N = poly_N.astype(np.float32)
            fd_meas = (fd_meas + poly_N[np.newaxis, :]).astype(np.float32)
            fd_diff = (fd_diff + poly_N[np.newaxis, :]).astype(np.float32)

        burst_t0 = b["start_time"] - t0_scene
        blk_lines = fvl + (np.arange(n_blk) + 0.5) * blsz
        per_burst.append({
            "meas": fd_meas, "model": fd_model, "diff": fd_diff,
            "rg_idx": rg_idx, "az_time": burst_t0 + blk_lines * ati,
        })

    if not per_burst:
        raise RuntimeError(f"No burst produced a Doppler block at blsz={blsz}")

    rg_idx_ref = max((p["rg_idx"] for p in per_burst), key=len)
    fd_measured = np.concatenate(
        [_resample_range(p["meas"], p["rg_idx"], rg_idx_ref) for p in per_burst], axis=0
    )
    fd_model = np.concatenate(
        [_resample_range(p["model"], p["rg_idx"], rg_idx_ref) for p in per_burst], axis=0
    )
    fd_diff = np.concatenate(
        [_resample_range(p["diff"], p["rg_idx"], rg_idx_ref) for p in per_burst], axis=0
    )
    az_time_s = np.concatenate([p["az_time"] for p in per_burst])
    range_m = r0 + rg_idx_ref * dr

    if (
        add_demod_back in ("blend", "hanning")
        and all("doppler_polynomial" in b for b in bursts)
    ):
        c_light = 299_792_458.0
        tau_ref = 2.0 * range_m / c_light
        poly_field = np.stack([
            np.asarray(
                sum(c * (tau_ref - b["doppler_srdelay"]) ** k
                    for k, c in enumerate(b["doppler_polynomial"])),
                dtype=np.float64,
            )
            for b in bursts
        ])

        if add_demod_back == "blend":
            burst_centers = np.array([
                (b["start_time"] - t0_scene)
                + 0.5 * (b["first_valid_line"] + b["last_valid_line"]) * ati
                for b in bursts
            ])
            smooth = np.empty_like(fd_measured)
            for rg in range(fd_measured.shape[1]):
                smooth[:, rg] = np.interp(
                    az_time_s, burst_centers, poly_field[:, rg]
                ).astype(np.float32)
        else:  # 'hanning'
            inter_burst_lines = int(round(
                (bursts[1]["start_time"] - bursts[0]["start_time"]) / ati
            ))
            overlap_n = max(0, lpb - inter_burst_lines)
            win = _burst_window(lpb, overlap_n)
            az_lines = az_time_s / ati
            az0 = np.array(
                [(b["start_time"] - t0_scene) / ati for b in bursts]
            )
            weights = np.zeros((len(az_lines), len(bursts)), dtype=np.float64)
            for j in range(len(bursts)):
                rel = az_lines - az0[j]
                inside = (rel >= 0) & (rel < lpb)
                if not inside.any():
                    continue
                ridx = np.clip(rel[inside].astype(int), 0, lpb - 1)
                weights[inside, j] = win[ridx] ** 2
            wsum = weights.sum(axis=1, keepdims=True)
            wsum = np.where(wsum > 0, wsum, 1.0)
            smooth = ((weights @ poly_field) / wsum).astype(np.float32)

        fd_measured = (fd_measured + smooth).astype(np.float32)
        fd_diff = (fd_diff + smooth).astype(np.float32)

    print(
        f"  computed {fd_measured.shape[0]} blocks x {fd_measured.shape[1]} "
        f"range cols ({len(per_burst)} bursts)"
    )
    return {
        "fd_measured": fd_measured,
        "fd_model": fd_model,
        "fd_diff": fd_diff,
        "range_m": range_m,
        "az_time_s": az_time_s,
        "blsz_lines": int(blsz),
        "mosaic_mode": "mosaic_last",
        "demod_mode": add_demod_back,
    }


def gamma_doppler_mosaic_last(
    blsz: int,
    gamma_dir: str = "data/sentinel-1/gamma_iw1",
    base_id: str = "20260205_iw1_vv",
    out_dir: str | None = None,
    overwrite: bool = False,
    add_demod_back: "str | bool" = "blend",
    return_dict: bool = False,
) -> "str | dict":
    """Estimate the Doppler centroid per deramped burst, then stitch in azimuth.

    ``doppler_2d_SLC`` is run once per burst over that burst's valid-line range
    of the deramped (burst-format) SLC ``{base_id}.deramp.slc``, so no
    estimation block ever straddles a burst boundary.  The per-burst Doppler
    grids are concatenated in azimuth order — GAMMA's ``SLC_mosaic_ScanSAR``
    only mosaics complex SLCs, not Doppler grids, so the stitch is a plain
    concatenation (the agreed "do it with our pipeline" fallback).

    Block-centre azimuth times are referenced to the scene start using the
    per-burst ``burst_start_time`` from the ``.tops_par``, so the output ``.npz``
    is directly comparable to :func:`gamma_doppler_mosaic_first`.

    Parameters / Returns: see :func:`gamma_doppler_mosaic_first`.

    If ``return_dict=True``, no ``.npz`` is written and the result dict is
    returned in memory (per-burst ``.txt`` files go to a tempdir that's
    deleted on return).  ``out_dir`` and ``overwrite`` are ignored.
    """
    add_demod_back = _normalise_demod_back(add_demod_back)

    deramp_slc = os.path.join(gamma_dir, f"{base_id}.deramp.slc")
    deramp_par = deramp_slc + ".par"
    tops_par = deramp_slc + ".tops_par"

    if not os.path.exists(deramp_slc):
        raise FileNotFoundError(
            f"Deramped burst SLC not found: {deramp_slc}\n"
            f"  Run shell_commands/gamma_iw1_deramp_merge_doppler.sh first,"
            f" or use run_gamma_pipeline_from_safe / gamma_prep_scene."
        )

    if return_dict:
        with tempfile.TemporaryDirectory(prefix="gamma_dop2d_") as td:
            return _compute_doppler_mosaic_last(
                blsz, deramp_slc, deramp_par, tops_par,
                add_demod_back, txt_dir=td,
                txt_prefix=f"{base_id}.dop2d.mosaic_last.blsz{blsz}",
            )

    out_dir = out_dir or gamma_dir
    suffix = "" if add_demod_back == "none" else f"_demod_{add_demod_back}"
    npz = os.path.join(out_dir, f"{base_id}.dop2d.mosaic_last{suffix}.blsz{blsz}.npz")
    if os.path.exists(npz) and not overwrite:
        print(f"  [cached] {npz}")
        return npz

    result = _compute_doppler_mosaic_last(
        blsz, deramp_slc, deramp_par, tops_par,
        add_demod_back, txt_dir=out_dir,
        txt_prefix=f"{base_id}.dop2d.mosaic_last.blsz{blsz}",
    )

    np.savez(
        npz,
        fd_measured=result["fd_measured"],
        fd_model=result["fd_model"],
        fd_diff=result["fd_diff"],
        range_m=result["range_m"],
        az_time_s=result["az_time_s"],
        blsz_lines=np.array(blsz),
        mosaic_mode=np.array("mosaic_last"),
        demod_mode=np.array(add_demod_back),
    )
    print(f"  saved -> {npz}")
    return npz
