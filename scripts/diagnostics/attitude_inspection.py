"""Inspect Sentinel-1 attitude quaternion conventions for mispointing analysis."""

import numpy as np

from scripts.sentinel_1.safe_io import find_safe_files, parse_annotation
from scripts.sentinel_1.aux_files import apply_poeorb
from scripts.sentinel_1.rvl import (
    _interpolate_orbit, _interpolate_attitude_quat, _quat_to_matrix,
    _ecef_to_j2000, _gmst_rad, _OMEGA_E, _latlon_to_ecef
)

SLC    = 'data/sentinel-1/S1A_IW_SLC__1SDV_20260205T165251_20260205T165321_063086_07EAEF_2642.SAFE'
POEORB = 'data/sentinel-1/S1A_OPER_AUX_POEORB_OPOD_20260225T070420_V20260204T225942_20260206T005942.EOF'

files = find_safe_files(SLC, 'iw1', 'vv')
annot = apply_poeorb(parse_annotation(files['annotation']), POEORB)

t = annot.bursts[0].azimuth_time
q = _interpolate_attitude_quat(annot, t)
R_actual = _quat_to_matrix(*q)   # may be body→J2000 or J2000→body

pos_ecef, vel_ecef = _interpolate_orbit(annot, t)
gmst = _gmst_rad(t)
r_j = _ecef_to_j2000(pos_ecef, gmst)
v_j_rot = _ecef_to_j2000(vel_ecef, gmst)
omega_e = np.array([0., 0., _OMEGA_E])
v_j = v_j_rot + np.cross(omega_e, r_j)   # full inertial velocity

# LORF in J2000
z_lorf = -r_j / np.linalg.norm(r_j)   # nadir
x_lorf =  v_j / np.linalg.norm(v_j)   # along-track
y_lorf = np.cross(z_lorf, x_lorf); y_lorf /= np.linalg.norm(y_lorf)
R_LORF = np.column_stack([x_lorf, y_lorf, z_lorf])   # LORF→J2000

print('LORF x (along-track in J2000):', np.round(x_lorf, 4))
print('LORF y (orbit-normal in J2000):', np.round(y_lorf, 4))
print('LORF z (nadir in J2000):        ', np.round(z_lorf, 4))
print()

# The nominal look direction toward ground in J2000
# Use annotation center range (mid-sample) and annotation first-line lat/lon
# Approximate: take the satellite position minus a fraction toward Earth center
# Nominal look angle ~ 33° incidence for IW1 center range
eta_nom = np.radians(29.45)  # off-nadir angle from AUX_INS
# Look direction in LORF (nadir + right): [0, sin(eta), cos(eta)] if z_lorf=nadir
look_in_lorf_A = np.array([0., np.sin(eta_nom),  np.cos(eta_nom)])  # z=nadir convention
look_in_lorf_B = np.array([0., np.sin(eta_nom), -np.cos(eta_nom)])  # z=anti-nadir convention

look_j2000_A = R_LORF @ look_in_lorf_A  # look direction in J2000

# Try BOTH quaternion conventions:
# Convention 1: R_actual = body→J2000  (current assumption)
#   Body z in J2000 = R_actual[:,2] (column 2)
# Convention 2: R_actual = J2000→body
#   Body z in J2000 = R_actual[2,:] (row 2) = R_actual.T[:,2]

for name, R_b2j in [('R_actual (body→J2000)', R_actual),
                    ('R_actual.T (body→J2000)', R_actual.T)]:
    bz_j2000 = R_b2j[:, 2]   # body z axis in J2000 (column 2 = third axis)
    bz_lorf  = R_LORF.T @ bz_j2000
    angle_A = np.degrees(np.arccos(np.clip(np.dot(bz_j2000, look_j2000_A), -1, 1)))

    # Also check body y
    by_j2000 = R_b2j[:, 1]
    by_lorf  = R_LORF.T @ by_j2000

    print(f'--- {name} ---')
    print(f'  Body z in LORF: {np.round(bz_lorf, 3)}')
    print(f'  Body y in LORF: {np.round(by_lorf, 3)}')
    print(f'  Angle(body-z, nominal-look): {angle_A:.1f}°')
    # For a right-looking SAR, body y should be close to the look direction
    # (body y = right-side = look direction for traditional body frame)
    angle_by = np.degrees(np.arccos(np.clip(np.dot(by_j2000, look_j2000_A), -1, 1)))
    print(f'  Angle(body-y, nominal-look): {angle_by:.1f}°')
    print()
