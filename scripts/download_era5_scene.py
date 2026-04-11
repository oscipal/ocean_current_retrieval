#!/usr/bin/env python3
"""
Download ERA5 wind, wave and Stokes drift for the S1 scene area.

Scene: IW1, 2026-02-05 ~16:53 UTC
  lat [-36, -33], lon [26, 29]

Variables downloaded (all from reanalysis-era5-single-levels):
  u10, v10        10-m wind components          [m/s]
  ust, vst        Surface Stokes drift           [m/s]
  swh             Significant wave height        [m]
  mwp             Mean wave period               [s]
  mwd             Mean wave direction            [deg]
  mdww            Mean direction of wind waves   [deg]
  shww            Significant height wind waves  [m]

Output: data/era5_data/era5_scene_20260205.nc

Usage
-----
  python scripts/download_era5_scene.py

Requires a valid ~/.cdsapirc with your CDS UID and API key.
See: https://cds.climate.copernicus.eu/api-how-to
"""

import cdsapi

OUT = 'data/era5_data/era5_scene_20260205.nc'

c = cdsapi.Client()

c.retrieve(
    'reanalysis-era5-single-levels',
    {
        'product_type': 'reanalysis',
        'variable': [
            '10m_u_component_of_wind',
            '10m_v_component_of_wind',
            'u_component_stokes_drift',
            'v_component_stokes_drift',
            'significant_height_of_combined_wind_waves_and_swell',
            'mean_wave_period',
            'mean_wave_direction',
            'mean_direction_of_wind_waves',
            'significant_height_of_wind_waves',
        ],
        'year':  '2026',
        'month': '02',
        'day':   '05',
        'time':  ['16:00', '17:00'],
        # CDS area format: [North, West, South, East]
        'area':  [-33, 26, -36, 29],
        'format': 'netcdf',
    },
    OUT,
)

print(f'Saved → {OUT}')
