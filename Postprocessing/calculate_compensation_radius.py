#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
calculate_compensation_radius.py
---------------------------------
Reads the processed shell-profile HDF5 files for a hydrodynamic simulation and
its dark-matter-only (DMO) counterpart, computes the compensation radius (the
halocentric radius at which the cumulative total mass in the hydro run equals the
cumulative DMO mass), and writes the results to a compact HDF5 file.

The compensation radius is only meaningful for hydro/DMO pairs, so this script
requires BOTH a hydro and a DMO profile set.

IMPORTANT — index-matching assumption
    This script assumes the hydro and DMO halo catalogues are **index-matched**
    (hydro row i corresponds to DMO row i), i.e. the DMO run shares the hydro
    FoF halo ordering (the usual ``subfiles_order`` setup).  If the two
    catalogues are **not** index-matched, the matching must be done as a
    *preprocessing* step: reorder (or subset) the profiles and catalogues so
    that row i of the hydro arrays corresponds to row i of the DMO arrays
    *before* running this script.

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
    calculate_compensation_radius,
    save_dict_to_hdf5,
    add_header_attrs,
    print_memory_mb,
)

t0 = time.time()
print_memory_mb()

# ── Configuration ─────────────────────────────────────────────────────────────

# SimulationLabel as set in the PROFI parameter file — hydro and DMO separately.
simulation_label     = 'MySim'
simulation_label_DMO = 'MySim_DMO'

# Paths to the PROFI OutputDir — the script looks for processed profiles under
# <profile_base_path[_DMO]>/snap<N>/Group_Profiles_<sim_name>_snap<N>_Processed.hdf5
profile_base_path     = '/path/to/profi/output/MySim_PROFI_10R200_nbins100_minMvir0.1_subfiles_order'
profile_base_path_DMO = '/path/to/profi/output/MySim_DMO_PROFI_10R200_nbins100_minMvir0.1_subfiles_order'

# Snapshots to process.
snapnum_list        = [99]

# Number of parallel workers (one per snapshot).
n_workers           = len(snapnum_list)

# Cosmological parameters — must match those used in PROFI.
cosmo_h             = 0.6774
cosmo_Omega_m       = 0.3089
cosmo_Omega_b       = 0.0486

# ── Compensation radius parameters ────────────────────────────────────────────

# Minimum search radius in units of R200c (hydro).
r_comp_normalized_minlim = 0.5

# Fractional tolerance on M_hydro/M_DMO.  A bin counts as compensated when the
# ratio lies within [1 - error_r, 1 + error_r].
error_r                  = 0.0093

# ── Derived constants (do not edit) ──────────────────────────────────────────
# Critical density at H0 = 100 km/s/Mpc in [1e10 Msun h^2 / Mpc^3]
_RHO_CRIT_100 = 27.7536627
rho_mean_arepo = _RHO_CRIT_100 * cosmo_Omega_m / 1e9   # [1e10 Msun h^2 / kpc^3]


# ── Halo catalogue loader (required) ─────────────────────────────────────────

def load_halo_catalogue(snapnum):
    """
    Return a dict with the hydro and DMO halo properties for `snapnum`:
        'Group_M_Crit200'     — hydro halo masses  [1e10 Msun/h]
        'Group_R_Crit200'     — hydro halo radii   [kpc/h]
        'Group_M_Crit200_DMO' — DMO   halo masses  [1e10 Msun/h]
        'Group_R_Crit200_DMO' — DMO   halo radii   [kpc/h]

    Hydro row i is assumed to correspond to DMO row i (index-matched).  If the
    two catalogues are NOT index-matched, do the matching as a preprocessing
    step and return arrays that are already aligned row-by-row.

    Implement this function to read the FOF catalogues from your simulation.

    Example using h5py directly (adapt paths for your setup):
    Read subfiles in numeric index order to match the PROFI profile order.

        import glob, h5py, numpy as np
        def _read(fof_dir):
            files = sorted(
                glob.glob(f'{fof_dir}/fof_subhalo_tab_{snapnum:03d}.*.hdf5'),
                key=lambda fp: int(fp.rsplit('.', 2)[1]))
            m200, r200 = [], []
            for fp in files:
                with h5py.File(fp, 'r') as f:
                    m200.append(f['Group/Group_M_Crit200'][()])
                    r200.append(f['Group/Group_R_Crit200'][()])
            return np.concatenate(m200), np.concatenate(r200)

        m_h, r_h = _read(f'/path/to/hydro/output/groups_{snapnum:03d}')
        m_d, r_d = _read(f'/path/to/dmo/output/groups_{snapnum:03d}')
        return {'Group_M_Crit200': m_h, 'Group_R_Crit200': r_h,
                'Group_M_Crit200_DMO': m_d, 'Group_R_Crit200_DMO': r_d}
    """
    raise NotImplementedError(
        "load_halo_catalogue() is not implemented. "
        "Fill in the function body above for your simulation.")


# ── Per-snapshot worker ───────────────────────────────────────────────────────

def process_snapshot(snapnum):
    sim_name     = os.path.basename(simulation_label.rstrip('/'))
    sim_name_DMO = os.path.basename(simulation_label_DMO.rstrip('/'))

    snap_dir     = os.path.join(profile_base_path,     f'snap{snapnum}')
    snap_dir_DMO = os.path.join(profile_base_path_DMO, f'snap{snapnum}')

    profile_path     = os.path.join(
        snap_dir,
        f'Group_Profiles_{sim_name}_snap{snapnum}_Processed.hdf5')
    profile_path_DMO = os.path.join(
        snap_dir_DMO,
        f'Group_Profiles_{sim_name_DMO}_snap{snapnum}_Processed.hdf5')
    Rconv_path = os.path.join(
        snap_dir,
        f'Group_Compensation_Radius_{sim_name}_snap{snapnum}.hdf5')

    if not os.path.exists(profile_path):
        print(f'[SKIP] Hydro profile {profile_path} does not exist.')
        return snapnum, None
    if not os.path.exists(profile_path_DMO):
        print(f'[SKIP] DMO profile {profile_path_DMO} does not exist.')
        return snapnum, None

    print(f'\n[INFO] === Snapshot {snapnum} ===')
    print(f'[INFO] Hydro profile: {profile_path}')
    print(f'[INFO] DMO   profile: {profile_path_DMO}')

    # Load halo catalogues and filter by the mass threshold stored in the header
    halo = load_halo_catalogue(snapnum)

    # Read the mass threshold from the hydro profile header
    with h5py.File(profile_path, 'r') as f:
        M200_minlim = float(f['Header'].attrs['Group_M_Crit200_minlim'])
    print(f'[INFO] M200 threshold (from profile header): {M200_minlim:.6f}')

    valid_idx = np.where((halo['Group_M_Crit200'] > M200_minlim)
                         & (halo['Group_M_Crit200_DMO'] > M200_minlim))[0]
    halo = {k: v[valid_idx] for k, v in halo.items()}
    print(f'[INFO] Matched halos above threshold: {len(valid_idx)}')

    # Split the combined catalogue into the per-run dicts the helper expects
    halo_hydro = {'Group_M_Crit200': halo['Group_M_Crit200'],
                  'Group_R_Crit200': halo['Group_R_Crit200']}
    halo_dmo   = {'Group_M_Crit200': halo['Group_M_Crit200_DMO'],
                  'Group_R_Crit200': halo['Group_R_Crit200_DMO']}

    # Load the shell profiles (only the fields needed for R_comp)
    fields_needed = ['Radius', 'rho_cum', 'volume_cum']
    profiles     = open_hdf5(profile_path,     fields=fields_needed, print_header=True)
    profiles_DMO = open_hdf5(profile_path_DMO, fields=fields_needed)
    print_memory_mb()

    # Compute compensation radius (hydro-indexed output)
    data_Rconv = calculate_compensation_radius(
        profiles, profiles_DMO, halo_hydro, halo_dmo,
        rho_mean_arepo=rho_mean_arepo,
        r_comp_normalized_minlim=r_comp_normalized_minlim,
        error_r=error_r)

    n_valid = int(np.sum(~np.isnan(data_Rconv['R_comp'])))
    print(f'[INFO] Valid R_comp: {n_valid} / {len(data_Rconv["R_comp"])} hydro halos')

    # Write output
    print(f'[INFO] Writing {Rconv_path}')
    with h5py.File(Rconv_path, 'w') as fout:
        add_header_attrs(fout,
            keys=['simulation', 'snapnum',
                  'R_comp_minlim', 'R_comp_error_r', 'index_convention', 'contact'],
            values=[sim_name, snapnum,
                    r_comp_normalized_minlim, error_r, 'hydro',
                    'M. Ayromlou (ayromlou@gmail.com)'])
        save_dict_to_hdf5(fout, data_Rconv)
    print(f'[INFO] Done: {Rconv_path}')

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
