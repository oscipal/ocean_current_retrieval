import sys, os
sys.path.insert(0, '.')
from scripts.rvl_pipeline import diagnose_mispointing_aux_ins

SLC     = 'data/sentinel-1/S1A_IW_SLC__1SDV_20260205T165251_20260205T165321_063086_07EAEF_2642.SAFE'
AUX_INS = 'data/sentinel-1/S1A_AUX_INS_V20250601T000000_G20251015T084917.SAFE'
POEORB  = 'data/sentinel-1/S1A_OPER_AUX_POEORB_OPOD_20260225T070420_V20260204T225942_20260206T005942.EOF'

diagnose_mispointing_aux_ins(
    SLC, 'iw1', AUX_INS,
    poeorb_path=POEORB,
    out_path='plots/mispointing_aux_ins.png',
)
