#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate_closure_radius.py
----------------------------
Reads the processed shell-profile HDF5 files produced by
process_profiles_to_shells.py, computes the closure radius and baryon
fraction for each halo, and writes the results to a compact HDF5 file.

Dependencies: numpy, h5py (standard scientific Python; no additional packages needed).

Written by M. Ayromlou
"""

import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import h5py

# Import helpers from the same directory
sys.path.insert(0, os.path.dirname(__file__))
from profi_helpers import (
    open_hdf5,
    calculate_closure_radius,
    calculate_baryon_fraction_at_radius,
    save_dict_to_hdf5,
    add_header_attrs,
    print_memory_mb,
)

t0 = time.time()
print_memory_mb()

# ── Configuration ─────────────────────────────────────────────────────────────

# SimulationLabel as set in the PROFI parameter file.
simulation_label    = 'MySim'

# Path to the PROFI OutputDir — the script looks for processed profiles under
# <profile_base_path>/snap<N>/Group_Profiles_<sim_name>_snap<N>_Processed.hdf5
profile_base_path   = '/path/to/profi/output/MySim_PROFI_10R200_nbins100_minMvir0.1_subfiles_order'

# Snapshots to process.
snapnum_list        = [99]

# Number of parallel workers (one per snapshot).
n_workers           = len(snapnum_list)

# Cosmological parameters — must match those used in PROFI.
cosmo_h             = 0.6774
cosmo_Omega_m       = 0.3089
cosmo_Omega_b       = 0.0486

# ── Closure radius parameters ─────────────────────────────────────────────────

# Minimum closure radius in units of R200c.
r_cl_normalized_minlim   = 0.5

# Fallback radius in units of R200c when no bin meets the closure conditions.
# None uses the profile's outer edge; eligible closure bins take precedence.
r_cl_normalized_maxlim   = None

# Fractional tolerance on f_b used to identify the closure radius.
# A bin counts if:  f_b * (1 - tol) <= baryon_fraction <= f_b * (1 + tol)
error_fb_planck_percent  = 0.05   # 5%

# ── Derived constants (do not edit) ──────────────────────────────────────────
f_b = cosmo_Omega_b / cosmo_Omega_m


# ── Halo catalogue loader (required) ─────────────────────────────────────────

def load_halo_catalogue(snapnum):
    """
    Return a dict with at least the following keys for `snapnum`:
        'Group_M_Crit200'  — halo masses  [1e10 Msun/h]
        'Group_R_Crit200'  — halo radii   [kpc/h]

    Implement this function to read the FOF catalogue from your simulation.

    Example using h5py directly (adapt paths for your setup):
    Read subfiles in numeric index order to match the PROFI profile order.

        import glob, h5py, numpy as np
        fof_dir = f'/path/to/simulation/output/groups_{snapnum:03d}'
        files = sorted(
            glob.glob(f'{fof_dir}/fof_subhalo_tab_{snapnum:03d}.*.hdf5'),
            key=lambda fp: int(fp.rsplit('.', 2)[1]))
        m200, r200 = [], []
        for fp in files:
            with h5py.File(fp, 'r') as f:
                m200.append(f['Group/Group_M_Crit200'][()])
                r200.append(f['Group/Group_R_Crit200'][()])
        return {'Group_M_Crit200': np.concatenate(m200),
                'Group_R_Crit200': np.concatenate(r200)}
    """
    raise NotImplementedError(
        "load_halo_catalogue() is not implemented. "
        "Fill in the function body above for your simulation.")


# ── Per-snapshot worker ───────────────────────────────────────────────────────

def process_snapshot(snapnum):
    sim_name = os.path.basename(simulation_label.rstrip('/'))
    snap_dir = os.path.join(profile_base_path, f'snap{snapnum}')

    profile_path = os.path.join(
        snap_dir,
        f'Group_Profiles_{sim_name}_snap{snapnum}_Processed.hdf5')
    Rc_path = os.path.join(
        snap_dir,
        f'Group_Closure_Radius_{sim_name}_snap{snapnum}.hdf5')

    if not os.path.exists(profile_path):
        print(f'[SKIP] {profile_path} does not exist.')
        return snapnum, None

    print(f'\n[INFO] === Snapshot {snapnum} ===')
    print(f'[INFO] Profile: {profile_path}')

    # Load halo catalogue and filter by mass threshold stored in profile header
    halo = load_halo_catalogue(snapnum)

    # Read the mass threshold from the processed profile header
    with h5py.File(profile_path, 'r') as f:
        M200_minlim = float(f['Header'].attrs['Group_M_Crit200_minlim'])
    print(f'[INFO] M200 threshold (from profile header): {M200_minlim:.6f}')

    valid_idx = np.where(halo['Group_M_Crit200'] > M200_minlim)[0]
    halo = {k: v[valid_idx] for k, v in halo.items()}
    print(f'[INFO] Halos above threshold: {len(valid_idx)}')

    # Load the shell profiles (only the fields needed for closure radius)
    fields_needed = ['Radius', 'rho', 'rho_cum']
    profiles = open_hdf5(profile_path, fields=fields_needed, print_header=True)
    print_memory_mb()

    # Compute closure radius
    data_Rc = calculate_closure_radius(
        profiles, halo, f_b=f_b,
        r_cl_normalized_minlim=r_cl_normalized_minlim,
        r_cl_normalized_maxlim=r_cl_normalized_maxlim,
        error_fb_planck_percent=error_fb_planck_percent)

    # Baryon fraction at R200c
    data_Rc['f_b_R200c'] = calculate_baryon_fraction_at_radius(
        profiles, halo, f_b=f_b, radius_in_R200c=1.0)

    # Attach halo properties for convenience
    data_Rc['Group_M_Crit200'] = halo['Group_M_Crit200'].astype(np.float32)
    data_Rc['Group_R_Crit200'] = halo['Group_R_Crit200'].astype(np.float32)

    # Write output
    print(f'[INFO] Writing {Rc_path}')
    with h5py.File(Rc_path, 'w') as fout:
        add_header_attrs(fout,
            keys=['simulation', 'snapnum',
                  'R_cl_minlim', 'error_fb_planck_percent', 'contact'],
            values=[sim_name, snapnum,
                    r_cl_normalized_minlim, error_fb_planck_percent,
                    'M. Ayromlou (ayromlou@gmail.com)'])
        save_dict_to_hdf5(fout, data_Rc)
    print(f'[INFO] Done: {Rc_path}')

    return snapnum, True


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    if n_workers > 1 and len(snapnum_list) > 1:
        print(f'[INFO] Processing {len(snapnum_list)} snapshots '
              f'with {n_workers} parallel workers.')
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = {ex.submit(process_snapshot, s): s for s in snapnum_list}
            for fut in as_completed(futures):
                sn, ok = fut.result()
                print(f'[INFO] Snapshot {sn} '
                      f'{"completed" if ok else "skipped"}.')
    else:
        for sn in snapnum_list:
            process_snapshot(sn)

    print(f'\n[INFO] All done. Total time: {time.time() - t0:.1f} s')
