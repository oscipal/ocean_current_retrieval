#!/usr/bin/env python3
"""Download ERA5 wind and wave fields from a JSON configuration file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cdsapi


WIND_VARIABLES = [
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
]

WAVE_VARIABLES = [
    "u_component_stokes_drift",
    "v_component_stokes_drift",
    "significant_height_of_combined_wind_waves_and_swell",
    "mean_wave_period",
    "mean_wave_direction",
    "mean_direction_of_wind_waves",
    "significant_height_of_wind_waves",
]


def _load_config(path: str) -> dict:
    with open(path, "r", encoding="ascii") as f:
        config = json.load(f)

    required = ["download_dir", "year", "month", "day", "time", "area"]
    missing = [key for key in required if key not in config]
    if missing:
        raise ValueError(f"config missing required keys: {', '.join(missing)}")

    if not isinstance(config["time"], list) or not config["time"]:
        raise ValueError("config['time'] must be a non-empty list like ['17:00']")
    if not isinstance(config["area"], list) or len(config["area"]) != 4:
        raise ValueError("config['area'] must be [north, west, south, east]")

    return config


def _request_body(config: dict, variables: list[str]) -> dict:
    return {
        "product_type": "reanalysis",
        "variable": variables,
        "year": str(config["year"]),
        "month": str(config["month"]).zfill(2),
        "day": str(config["day"]).zfill(2),
        "time": config["time"],
        "area": config["area"],
        "format": "netcdf",
    }


def run(config_path: str) -> None:
    config = _load_config(config_path)
    out_dir = Path(config["download_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    wind_path = out_dir / "era5_wind.nc"
    wave_path = out_dir / "era5_wave.nc"

    client = cdsapi.Client()

    print(f"Downloading ERA5 wind to {wind_path} ...")
    client.retrieve(
        "reanalysis-era5-single-levels",
        _request_body(config, WIND_VARIABLES),
        str(wind_path),
    )

    print(f"Downloading ERA5 wave/Stokes to {wave_path} ...")
    client.retrieve(
        "reanalysis-era5-single-levels",
        _request_body(config, WAVE_VARIABLES),
        str(wave_path),
    )

    print(f"Saved wind  -> {wind_path}")
    print(f"Saved wave  -> {wave_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download ERA5 wind and wave products from a JSON config.",
    )
    parser.add_argument(
        "--config",
        default="config/download_era5.json",
        help="Path to JSON config file.",
    )
    args = parser.parse_args()
    run(args.config)


if __name__ == "__main__":
    main()
