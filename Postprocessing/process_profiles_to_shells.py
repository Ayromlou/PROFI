#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
process_profiles_to_shells.py
------------------------------
Reads PROFI HDF5 cumulative profile files, converts them to shell (differential)
profiles, and saves the result to a single processed HDF5 file per snapshot.

Dependencies: numpy, h5py, concurrent.futures (all standard in scientific Python).
Optional: psutil (for memory reporting only).

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
    load_profi_profiles,
    make_shell_profiles,
    save_dict_to_hdf5,
    add_header_attrs,
    print_memory_mb,
)

t0 = time.time()
print_memory_mb()

# ── Configuration ─────────────────────────────────────────────────────────────
# All settings below must be consistent with the PROFI parameter file used to
# generate the profiles.

# SimulationLabel as set in the PROFI parameter file.
# Only the last path component is used for file-name matching, so
# 'MyOrg/MySim' and 'MySim' both match files named profile_MySim_snap*_*.h5.
simulation_label    = 'MySim'

# Path to the OutputDir used in the PROFI parameter file.
# The script looks for profiles under <profile_base_path>/snap<N>/.
profile_base_path   = '/path/to/profi/output/MySim_PROFI_10R200_nbins100_minMvir0.1_subfiles_order'

# Snapshots to process.
snapnum_list        = [99]

# Profile parameters — must match the PROFI parameter file.
nbins_profile       = 100    # nbins_profile

# n_partType_profiles: leave as None to auto-detect from the profile file,
# or set explicitly (1 = DMO, 2 = gas+DM, 3 = gas+DM+stars, …).
n_partType_profiles = None

# Number of parallel workers (one per snapshot).
# Set to 1 to run serially (safer for debugging or low-memory machines).
n_workers           = len(snapnum_list)

# Cosmological parameters — must match those used in PROFI.
# Defaults are Planck 2015; change if your simulation uses different values.
cosmo_h             = 0.6774
cosmo_Omega_m       = 0.3089
cosmo_Omega_b       = 0.0486

# shrink_profiles: if True, also save Subhalo_idx for downstream halo matching.
# Enabling this requires loading the halo catalogue from the FOF subhalo files.
# Implement load_halo_catalogue() below to read Group_M_Crit200 from your data.
shrink_profiles     = False

# ── Derived constants (do not edit) ──────────────────────────────────────────
# Critical density at H0 = 100 km/s/Mpc in [1e10 Msun h^2 / Mpc^3]
_RHO_CRIT_100 = 27.7536627
rho_mean_arepo = _RHO_CRIT_100 * cosmo_Omega_m / 1e9   # [1e10 Msun h^2 / kpc^3]
f_b            = cosmo_Omega_b / cosmo_Omega_m


# ── Optional: halo catalogue loader (needed for shrink_profiles only) ─────────

def load_halo_catalogue(snapnum):
    """
    Return a dict with at least ``{'Group_M_Crit200': array}`` for ``snapnum``.

    This function is only called when ``shrink_profiles = True``.
    Implement it to read the FOF catalogue from your simulation using h5py,
    yt, pynbody, or any other tool you prefer.

    Example using h5py directly (adapt paths for your simulation):
    Read subfiles in numeric index order to match the PROFI profile order.

        import glob, h5py, numpy as np
        fof_dir = f'/path/to/simulation/output/groups_{snapnum:03d}'
        files = sorted(
            glob.glob(f'{fof_dir}/fof_subhalo_tab_{snapnum:03d}.*.hdf5'),
            key=lambda fp: int(fp.rsplit('.', 2)[1]))
        m200 = []
        for fp in files:
            with h5py.File(fp, 'r') as f:
                m200.append(f['Group/Group_M_Crit200'][()])
        return {'Group_M_Crit200': np.concatenate(m200)}
    """
    raise NotImplementedError(
        "load_halo_catalogue() is not implemented. "
        "Either implement it for shrink_profiles=True, or set shrink_profiles=False.")


# ── Per-snapshot probe: read header attributes ────────────────────────────────

def _read_probe_attrs(fp):
    """Return (n_partType, n_metals, profi_M200_lim) from a probe subfile."""
    print(f'  [probe] {fp}')
    with h5py.File(fp, 'r') as f:
        n_pt  = int(f.attrs['n_partType'])
        n_met = int(f.attrs['n_metals'])
        if 'Group_M_Crit200_minlim' in f['Header'].attrs:
            mlim = float(f['Header'].attrs['Group_M_Crit200_minlim'])
        else:
            mlim = float(f.attrs['Group_M_Crit200_minlim'])
    return n_pt, n_met, mlim


# ── Per-snapshot worker ───────────────────────────────────────────────────────

def process_snapshot(snapnum):
    print(f'\n[INFO] === Snapshot {snapnum} ===')

    profile_path_snap = os.path.join(profile_base_path, f'snap{snapnum}')
    sim_name = os.path.basename(simulation_label.rstrip('/'))
    output_file = os.path.join(
        profile_path_snap,
        f'Group_Profiles_{sim_name}_snap{snapnum}_Processed.hdf5')
    print(f'[INFO] Output: {output_file}')

    if not os.path.exists(profile_path_snap):
        print(f'[SKIP] {profile_path_snap} does not exist.')
        return snapnum, False

    # Probe file: subfile 0 (always present for non-empty snapshots)
    probe = os.path.join(profile_path_snap,
                         f'profile_{sim_name}_snap{snapnum:03d}_0.h5')
    if not os.path.exists(probe):
        # Try auto-discovery
        import glob as _glob
        candidates = sorted(_glob.glob(
            os.path.join(profile_path_snap,
                         f'profile_*_snap{snapnum:03d}_0.h5')))
        if not candidates:
            print(f'[SKIP] No probe file found in {profile_path_snap}.')
            return snapnum, False
        probe = candidates[0]

    n_pt, n_met, profi_M200_lim = _read_probe_attrs(probe)
    n_partType = n_partType_profiles if n_partType_profiles is not None else n_pt
    print(f'[INFO] n_partType={n_partType}, n_metals={n_met}, '
          f'M200_lim={profi_M200_lim:.6f}')

    # Optional: load halo catalogue for Subhalo_idx matching
    if shrink_profiles:
        print('[INFO] Loading halo catalogue (shrink_profiles=True)…')
        halo_cat = load_halo_catalogue(snapnum)
        n_halos_cat = len(halo_cat['Group_M_Crit200'])
        print(f'[INFO] Catalogue: {n_halos_cat} halos')

    # Load cumulative profiles
    print(f'[INFO] Loading cumulative profiles from {profile_path_snap}…')
    profiles_cum, redshift, boxsize, is_hydro = load_profi_profiles(
        profile_path_snap, simulation_label, snapnum,
        n_partType=n_partType, nbins_profile=nbins_profile, n_metals=n_met)
    n_halos = profiles_cum['Radius'].shape[0]
    print(f'[INFO] Loaded {n_halos} halos, z={redshift:.3f}, '
          f'box={boxsize:.1f} kpc/h, is_hydro={is_hydro}')
    print_memory_mb()

    # Convert cumulative → shells
    print('[INFO] Converting to shell profiles…')
    profiles, meta = make_shell_profiles(
        profiles_cum, rho_mean_arepo, f_b,
        n_partType=n_partType, n_metals=n_met, is_hydro_sim=is_hydro)
    del profiles_cum
    print(f'[INFO] Shell profiles: {n_halos} halos × {meta["nbins"]} bins')
    print_memory_mb()

    # Optionally attach Subhalo_idx
    if shrink_profiles:
        valid_idx = np.where(
            halo_cat['Group_M_Crit200'] > profi_M200_lim)[0]
        if len(valid_idx) != n_halos:
            raise ValueError(
                f'Mass threshold mismatch: catalogue has {len(valid_idx)} halos '
                f'above {profi_M200_lim:.6f} but profiles have {n_halos} rows.')
        profiles['Subhalo_idx'] = valid_idx
        print(f'[INFO] Saved Subhalo_idx for {len(valid_idx)} halos.')

    # Write output
    print(f'[INFO] Writing {output_file}…')
    with h5py.File(output_file, 'w') as fout:
        add_header_attrs(fout,
            keys=['simulation', 'snapnum', 'nbins_profile',
                  'nbins_temperature', 'nbins_velocity',
                  'n_partType_profile', 'n_metals_profile',
                  'is_hydro_sim',
                  'Group_M_Crit200_minlim', 'contact'],
            values=[sim_name, snapnum, meta['nbins'],
                    meta['nbins_temp'], meta['nbins_vel'],
                    n_partType, meta['n_metals'],
                    meta['is_hydro_sim'],
                    profi_M200_lim, 'M. Ayromlou (ayromlou@gmail.com)'])
        save_dict_to_hdf5(fout, profiles)
    print(f'[INFO] Done: {output_file}')
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
