import sys, os
sys.path.insert(0, '.')
from scripts.s1_aux import parse_aux_cal
from scripts.s1_rvl import compute_gamma_ambiguity, compute_sideband_bias
import numpy as np

AUX_CAL = 'data/sentinel-1/S1A_AUX_CAL_V20190228T092500_G20240327T102320.SAFE'
aap = parse_aux_cal(AUX_CAL, 'iw1', 'vv')

prf = 486.486
lam = 0.0555
v   = 7583.6

print('aap.angles_deg range:', aap.angles_deg[0], '...', aap.angles_deg[-1])
print('aap.gain_db[:5]:', aap.gain_db[:5])

# Check gain at ambiguity angles
for k in range(1, 4):
    f = k * prf
    g = aap.two_way_at_doppler_hz(np.array([f]), v, lam)[0]
    print(f'k={k}: two_way gain = {g:.10f}')

gamma = compute_gamma_ambiguity(aap, prf, v, lam)
fsb   = compute_sideband_bias(aap, 0.0, prf, v, lam)
print(f'\ngamma = {gamma}')
print(f'sideband bias = {fsb}')
