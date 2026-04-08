"""
Sentinel-1 Level-2 OCN SAFE reader.

This module reads the Level-2 OCN NetCDF measurement files directly instead of
reconstructing RVL / OWI / OSW from Level-1 SLC data.
"""

from __future__ import annotations

from pathlib import Path

import xarray as xr


_OCN_PREFIXES = ("owi", "osw", "rvl")
_RADAR_MODES = ("ew", "iw", "sm", "wv")
_POLARISATIONS = ("hh", "hv", "vh", "vv")
_AUX_COORD_NAMES = {
    "lat",
    "latitude",
    "lon",
    "longitude",
    "incidenceangle",
    "incidence_angle",
    "heading",
    "landflag",
    "mask",
    "quality",
    "time",
}


def _measurement_tokens(path: str | Path) -> dict[str, str | None]:
    parts = Path(path).stem.lower().split("-")
    return {
        "component": next((part for part in parts if part in _OCN_PREFIXES), None),
        "family": next((part for part in parts if part in _RADAR_MODES), None),
        "swath": next(
            (
                part
                for part in parts
                if len(part) >= 3 and part[:2] in _RADAR_MODES and part[2:].isdigit()
            ),
            None,
        ),
        "polarisation": next((part for part in parts if part in _POLARISATIONS), None),
        "is_combined": "ocn" in parts,
    }


def _swath_matches(requested_swath: str | None, metadata: dict[str, str | None]) -> bool:
    if requested_swath is None:
        return True

    if metadata["swath"] is not None:
        return metadata["swath"] == requested_swath

    if metadata["is_combined"] and metadata["family"] is not None:
        return requested_swath.startswith(metadata["family"])

    return False


def _swath_index(swath: str) -> int:
    suffix = swath[2:]
    if not suffix.isdigit():
        raise ValueError(f"Unsupported swath selector: {swath}")
    index = int(suffix) - 1
    if index < 0:
        raise ValueError(f"Unsupported swath selector: {swath}")
    return index


def find_ocn_measurements(
    safe_dir: str,
    swath: str | None = None,
    polarisation: str | None = None,
) -> list[Path]:
    """
    Return Level-2 OCN measurement NetCDF files inside a SAFE directory.

    Parameters
    ----------
    safe_dir : str
        Path to the OCN SAFE directory.
    swath : str, optional
        Filter by swath string such as ``iw1`` or ``wv1``.
    polarisation : str, optional
        Filter by polarisation string such as ``vv`` or ``vh``.
    """
    measurement_dir = Path(safe_dir) / "measurement"
    if not measurement_dir.is_dir():
        raise FileNotFoundError(f"Missing measurement directory in {safe_dir}")

    swath_token = swath.lower() if swath else None
    pol_token = polarisation.lower() if polarisation else None

    matches: list[Path] = []
    for path in sorted(measurement_dir.glob("*.nc")):
        metadata = _measurement_tokens(path)
        if not (metadata["is_combined"] or metadata["component"] in _OCN_PREFIXES):
            continue
        if swath_token and not _swath_matches(swath_token, metadata):
            continue
        if pol_token and metadata["polarisation"] != pol_token:
            continue
        matches.append(path)

    if not matches:
        details = []
        if swath_token:
            details.append(f"swath={swath_token}")
        if pol_token:
            details.append(f"polarisation={pol_token}")
        suffix = f" ({', '.join(details)})" if details else ""
        raise FileNotFoundError(f"No OCN NetCDF measurement files found in {measurement_dir}{suffix}")

    return matches


def open_ocn_measurement(path: str | Path, decode_cf: bool = True) -> xr.Dataset:
    """
    Open a single Sentinel-1 OCN NetCDF measurement file.
    """
    dataset = xr.open_dataset(path, decode_cf=decode_cf).load()
    dataset.attrs["source_file"] = Path(path).name
    dataset.attrs["source_path"] = str(Path(path))
    return dataset


def available_ocn_components(dataset: xr.Dataset) -> list[str]:
    """
    Return the OCN components present in an opened NetCDF file.
    """
    present: list[str] = []
    lower_names = {name.lower() for name in dataset.variables}
    for prefix in _OCN_PREFIXES:
        if any(name.startswith(prefix) for name in lower_names):
            present.append(prefix)
    return present


def _variables_for_prefix(dataset: xr.Dataset, prefix: str) -> list[str]:
    prefix = prefix.lower()
    selected: list[str] = []
    for name in dataset.data_vars:
        lower_name = name.lower()
        if lower_name.startswith(prefix):
            selected.append(name)
            continue
        if lower_name in _AUX_COORD_NAMES:
            selected.append(name)
    return selected


def _promote_auxiliary_coords(dataset: xr.Dataset) -> xr.Dataset:
    coord_names = []
    for name, data in dataset.data_vars.items():
        if name.lower() not in _AUX_COORD_NAMES:
            continue
        if set(data.dims).issubset(dataset.dims):
            coord_names.append(name)
    if coord_names:
        dataset = dataset.set_coords(coord_names)
    return dataset


def extract_ocn_component(
    dataset: xr.Dataset,
    component: str,
    swath: str | None = None,
) -> xr.Dataset:
    """
    Extract one OCN component from a NetCDF measurement dataset.

    Parameters
    ----------
    dataset : xr.Dataset
        Opened OCN measurement dataset.
    component : str
        One of ``'owi'``, ``'osw'``, or ``'rvl'``.
    """
    prefix = component.lower()
    if prefix not in _OCN_PREFIXES:
        raise ValueError(f"Unsupported OCN component: {component}")

    component_vars = _variables_for_prefix(dataset, prefix)
    if not any(name.lower().startswith(prefix) for name in component_vars):
        raise KeyError(f"Component {component} not present in {dataset.attrs.get('source_file', 'dataset')}")

    extracted = dataset[component_vars]

    if swath:
        swath_dim = f"{prefix}Swath"
        if swath_dim in extracted.dims:
            extracted = extracted.isel({swath_dim: _swath_index(swath)}, drop=True)

    extra_coords = {
        name: dataset.coords[name]
        for name in dataset.coords
        if name.lower() in _AUX_COORD_NAMES and set(dataset.coords[name].dims).issubset(extracted.dims)
    }
    if extra_coords:
        extracted = extracted.assign_coords(extra_coords)
    extracted = _promote_auxiliary_coords(extracted)
    extracted.attrs.update(dataset.attrs)
    extracted.attrs["ocn_component"] = prefix
    if swath:
        extracted.attrs["selected_swath"] = swath
    return extracted


def _concat_component_datasets(component_datasets: list[xr.Dataset], source_names: list[str]) -> xr.Dataset:
    if len(component_datasets) == 1:
        dataset = component_datasets[0]
        dataset.attrs["source_files"] = list(source_names)
        return dataset

    stacked = []
    for name, dataset in zip(source_names, component_datasets, strict=False):
        stacked.append(dataset.expand_dims(source_file=[name]))
    merged = xr.concat(stacked, dim="source_file", combine_attrs="drop_conflicts")
    merged.attrs["source_files"] = list(source_names)
    return merged


def load_ocn_safe(
    safe_dir: str,
    swath: str | None = None,
    polarisation: str | None = None,
    decode_cf: bool = True,
) -> dict[str, xr.Dataset]:
    """
    Load direct Level-2 OCN parameters from a Sentinel-1 OCN SAFE product.

    Returns
    -------
    dict
        Keys are the available OCN components: ``'owi'``, ``'osw'``, ``'rvl'``.
        If multiple measurement files contain the same component, they are
        concatenated along a new ``source_file`` dimension.
    """
    measurement_paths = find_ocn_measurements(
        safe_dir,
        swath=swath,
        polarisation=polarisation,
    )

    grouped: dict[str, list[xr.Dataset]] = {prefix: [] for prefix in _OCN_PREFIXES}
    source_names: dict[str, list[str]] = {prefix: [] for prefix in _OCN_PREFIXES}

    for path in measurement_paths:
        dataset = open_ocn_measurement(path, decode_cf=decode_cf)
        for component in available_ocn_components(dataset):
            grouped[component].append(extract_ocn_component(dataset, component, swath=swath))
            source_names[component].append(path.name)

    loaded = {
        component: _concat_component_datasets(datasets, source_names[component])
        for component, datasets in grouped.items()
        if datasets
    }
    if not loaded:
        raise ValueError(f"No OCN components found in {safe_dir}")
    return loaded


def load_rvl_from_ocn(
    safe_dir: str,
    swath: str | None = None,
    polarisation: str | None = None,
    decode_cf: bool = True,
) -> xr.Dataset:
    """Convenience wrapper returning only the RVL dataset from an OCN SAFE."""
    return load_ocn_safe(
        safe_dir,
        swath=swath,
        polarisation=polarisation,
        decode_cf=decode_cf,
    )["rvl"]


def load_owi_from_ocn(
    safe_dir: str,
    swath: str | None = None,
    polarisation: str | None = None,
    decode_cf: bool = True,
) -> xr.Dataset:
    """Convenience wrapper returning only the OWI dataset from an OCN SAFE."""
    return load_ocn_safe(
        safe_dir,
        swath=swath,
        polarisation=polarisation,
        decode_cf=decode_cf,
    )["owi"]


def load_osw_from_ocn(
    safe_dir: str,
    swath: str | None = None,
    polarisation: str | None = None,
    decode_cf: bool = True,
) -> xr.Dataset:
    """Convenience wrapper returning only the OSW dataset from an OCN SAFE."""
    return load_ocn_safe(
        safe_dir,
        swath=swath,
        polarisation=polarisation,
        decode_cf=decode_cf,
    )["osw"]
