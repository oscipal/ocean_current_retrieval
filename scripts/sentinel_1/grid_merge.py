"""Merge burst-wise retrieval outputs onto common geocoded comparison grids."""

from __future__ import annotations

import numpy as np


def merge_burst_grids(
    results: list[dict],
    variable: str = "v_current_ocn",
    overlap: str = "average",
    resolution_deg: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Merge per-burst results onto a single regular lat/lon grid."""
    from scipy.interpolate import griddata

    if overlap not in ("average", "first", "last", "best_rmse"):
        raise ValueError("overlap must be 'average', 'first', 'last', or 'best_rmse'")

    all_lat = np.concatenate([r["lat"].ravel() for r in results])
    all_lon = np.concatenate([r["lon"].ravel() for r in results])
    valid = np.isfinite(all_lat) & np.isfinite(all_lon)
    grid_lat = np.arange(all_lat[valid].min(), all_lat[valid].max() + resolution_deg, resolution_deg)
    grid_lon = np.arange(all_lon[valid].min(), all_lon[valid].max() + resolution_deg, resolution_deg)
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)

    n_results = len(results)
    if overlap == "average":
        all_data = np.concatenate([r[variable].ravel() for r in results])
        ok = np.isfinite(all_lat) & np.isfinite(all_lon) & np.isfinite(all_data)
        merged = griddata(
            points=(all_lon[ok], all_lat[ok]),
            values=all_data[ok],
            xi=(mesh_lon, mesh_lat),
            method="linear",
        ).astype(np.float32)
        return grid_lat, grid_lon, merged

    layers = np.full((n_results, len(grid_lat), len(grid_lon)), np.nan, dtype=np.float32)
    for i, result in enumerate(results):
        lat_f = result["lat"].ravel()
        lon_f = result["lon"].ravel()
        data_f = result[variable].ravel()
        ok = np.isfinite(lat_f) & np.isfinite(lon_f) & np.isfinite(data_f)
        if ok.sum() < 3:
            continue
        layers[i] = griddata(
            points=(lon_f[ok], lat_f[ok]),
            values=data_f[ok],
            xi=(mesh_lon, mesh_lat),
            method="linear",
        ).astype(np.float32)

    valid_mask = np.isfinite(layers)
    if overlap == "first":
        merged = np.full((len(grid_lat), len(grid_lon)), np.nan, dtype=np.float32)
        for i in range(n_results):
            fill = valid_mask[i] & ~np.isfinite(merged)
            merged[fill] = layers[i][fill]
    elif overlap == "last":
        merged = np.full((len(grid_lat), len(grid_lon)), np.nan, dtype=np.float32)
        for i in range(n_results):
            merged[valid_mask[i]] = layers[i][valid_mask[i]]
    else:
        rmse_vals = np.array([r.get("rmse_vs_glo12", np.inf) for r in results])
        order = np.argsort(rmse_vals)[::-1]
        merged = np.full((len(grid_lat), len(grid_lon)), np.nan, dtype=np.float32)
        for i in order:
            merged[valid_mask[i]] = layers[i][valid_mask[i]]

    gaps = ~np.isfinite(merged)
    if gaps.any():
        all_data = np.concatenate([r[variable].ravel() for r in results])
        ok = np.isfinite(all_lat) & np.isfinite(all_lon) & np.isfinite(all_data)
        combined = griddata(
            points=(all_lon[ok], all_lat[ok]),
            values=all_data[ok],
            xi=(mesh_lon[gaps], mesh_lat[gaps]),
            method="linear",
        ).astype(np.float32)
        merged[gaps] = combined

    return grid_lat, grid_lon, merged


def merge_model_grid(
    results: list[dict],
    resolution_deg: float = 0.01,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Interpolate GLO12 model values from all bursts onto a common grid."""
    from scipy.interpolate import griddata

    all_lat = np.concatenate([r["lat"].ravel() for r in results])
    all_lon = np.concatenate([r["lon"].ravel() for r in results])
    all_data = np.concatenate([r["v_model"].ravel() for r in results])

    ok = np.isfinite(all_lat) & np.isfinite(all_lon) & np.isfinite(all_data)
    grid_lat = np.arange(all_lat[ok].min(), all_lat[ok].max() + resolution_deg, resolution_deg)
    grid_lon = np.arange(all_lon[ok].min(), all_lon[ok].max() + resolution_deg, resolution_deg)
    mesh_lon, mesh_lat = np.meshgrid(grid_lon, grid_lat)

    grid_model = griddata(
        points=(all_lon[ok], all_lat[ok]),
        values=all_data[ok],
        xi=(mesh_lon, mesh_lat),
        method="linear",
    ).astype(np.float32)
    return grid_lat, grid_lon, grid_model


def write_burst_kml(
    results: list[dict],
    out_path: str = "burst_polygons.kml",
) -> None:
    """Write a KML file with one polygon per burst."""
    import colorsys

    def _border(lat, lon):
        top = (lon[0, :], lat[0, :])
        right = (lon[1:, -1], lat[1:, -1])
        bottom = (lon[-1, -2::-1], lat[-1, -2::-1])
        left = (lon[-2:0:-1, 0], lat[-2:0:-1, 0])
        lons = np.concatenate([top[0], right[0], bottom[0], left[0], [top[0][0]]])
        lats = np.concatenate([top[1], right[1], bottom[1], left[1], [top[1][0]]])
        return lons, lats

    n_results = len(results)
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<kml xmlns="http://www.opengis.net/kml/2.2">',
        "<Document>",
        f'  <name>Burst polygons ({results[0]["subswath"].upper()})</name>',
    ]
    for i, result in enumerate(results):
        frac = i / max(n_results - 1, 1)
        h, s, v = (1.0 - frac) * 0.67, 1.0, 1.0
        rgb = colorsys.hsv_to_rgb(h, s, v)
        kml_col = f'88{int(rgb[2]*255):02x}{int(rgb[1]*255):02x}{int(rgb[0]*255):02x}'
        kml_line = f'ff{int(rgb[2]*255):02x}{int(rgb[1]*255):02x}{int(rgb[0]*255):02x}'
        style_id = f'burst{result["burst_idx"]}'
        lines += [
            f'  <Style id="{style_id}">',
            f"    <LineStyle><color>{kml_line}</color><width>2</width></LineStyle>",
            f"    <PolyStyle><color>{kml_col}</color></PolyStyle>",
            "  </Style>",
        ]

        lons, lats = _border(result["lat"], result["lon"])
        coords = " ".join(f"{lo:.6f},{la:.6f},0" for lo, la in zip(lons, lats))
        lines += [
            "  <Placemark>",
            f'    <name>Burst {result["burst_idx"]}</name>',
            f'    <description>bias={result["bias_vs_glo12"]:+.4f} m/s  RMSE={result["rmse_vs_glo12"]:.4f} m/s  r={result["r_vs_glo12"]:.4f}</description>',
            f"    <styleUrl>#{style_id}</styleUrl>",
            "    <Polygon><outerBoundaryIs><LinearRing>",
            f"      <coordinates>{coords}</coordinates>",
            "    </LinearRing></outerBoundaryIs></Polygon>",
            "  </Placemark>",
        ]

    lines += ["</Document>", "</kml>"]
    with open(out_path, "w", encoding="ascii") as f:
        f.write("\n".join(lines))


def compute_stats(sar: np.ndarray, model: np.ndarray) -> tuple[float, float, float]:
    """Return bias, RMSE, and correlation for finite overlapping cells."""
    mask = np.isfinite(sar) & np.isfinite(model)
    if mask.sum() < 2:
        return float("nan"), float("nan"), float("nan")
    diff = sar[mask] - model[mask]
    bias = float(np.mean(diff))
    rmse = float(np.sqrt(np.mean(diff**2)))
    corr = float(np.corrcoef(sar[mask], model[mask])[0, 1])
    return bias, rmse, corr
