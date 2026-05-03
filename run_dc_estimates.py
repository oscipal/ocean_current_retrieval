import sys, os
sys.path.insert(0, '.')
from scripts.rvl_pipeline import plot_dc_estimates

SLC_SAFE = 'data/sentinel-1/S1A_IW_SLC__1SDV_20260205T165251_20260205T165321_063086_07EAEF_2642.SAFE'
plot_dc_estimates(SLC_SAFE, 'iw1', out_path='plots/dc_estimates.png')
