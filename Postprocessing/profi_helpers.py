#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
profi_helpers.py
----------------
Helper functions for reading PROFI HDF5 profile files and converting
cumulative profiles to shell (differential) profiles.

No external dependencies beyond numpy, h5py, glob, os, and (optionally) psutil.

Written by M. Ayromlou
"""

import os
import re
import glob
import numpy as np
import h5py

# kpc → km conversion (exact, NIST / IAU 2012)
_KPC_TO_KM = 3.085677581e16


# ── Structured-array dtype ───────────────────────────────────────────────────

def build_profi_dtype(n_partType, nbins_profile, n_metals, pos_dim=3):
    """
    Build the numpy structured dtype that matches the PROFI cumulative-profile
    array layout.

    Parameters
    ----------
    n_partType : int
        Number of particle types in the profile (1 = DMO, 3 = gas+DM+stars, …).
    nbins_profile : int
        Number of radial bins.
    n_metals : int
        Number of metal species (10 for TNG/EAGLE, 11 for SIMBA).
    pos_dim : int, optional
        Spatial dimensions (default 3).

    Returns
    -------
    dtype : np.dtype
    """
    s_AB = (n_partType, nbins_profile)
    s_D  = (nbins_profile,)
    s_E  = (n_metals,  nbins_profile)
    s_C  = (pos_dim, n_partType, nbins_profile)

    return np.dtype([
        # ── Family D — radius (scalar per bin, no particle-type axis) ──────
        ('Radius',                          np.float32, s_D),
        # ── Family A/B — per particle-type × per bin ──────────────────────
        ('np',                              np.int32,   s_AB),
        ('vrad_times_mass',                 np.float64, s_AB),
        ('vLOS_z_times_mass',               np.float64, s_AB),
        ('mass',                            np.float64, s_AB),
        ('mass_out',                        np.float64, s_AB),
        ('mass_out_fast1',                  np.float64, s_AB),
        ('mass_out_fast2',                  np.float64, s_AB),
        ('mass_out_fast3',                  np.float64, s_AB),
        ('np_out',                          np.int32,   s_AB),
        ('vrad_times_mass_out',             np.float64, s_AB),
        ('vrad_times_mass_out_fast1',       np.float64, s_AB),
        ('vrad_times_mass_out_fast2',       np.float64, s_AB),
        ('vrad_times_mass_out_fast3',       np.float64, s_AB),
        ('vLOS_z_times_mass_out',           np.float64, s_AB),
        # ── Family C — 3-D flow vectors (pos_dim × n_partType × nbins) ────
        ('vLOS_times_mass',                 np.float64, s_C),
        ('matter_flow_vec',                 np.float64, s_C),
        ('matter_outflow_vec',              np.float64, s_C),
        # ── Family D — gas-only (single particle-type axis, nbins) ────────
        ('temp',                            np.float64, s_D),
        ('entropy',                         np.float64, s_D),
        ('L_bol',                           np.float64, s_D),
        ('L_bol_times_mass',                np.float64, s_D),
        ('N_e',                             np.float64, s_D),
        ('N_H0',                            np.float64, s_D),
        ('mass_temp1',                      np.float64, s_D),
        ('mass_temp2',                      np.float64, s_D),
        ('mass_temp3',                      np.float64, s_D),
        ('mass_out_temp1',                  np.float64, s_D),
        ('mass_out_temp2',                  np.float64, s_D),
        ('mass_out_temp3',                  np.float64, s_D),
        ('temp_out',                        np.float64, s_D),
        ('temp_out_fast1',                  np.float64, s_D),
        ('temp_out_fast2',                  np.float64, s_D),
        ('temp_out_fast3',                  np.float64, s_D),
        # legacy aliases (same data as mass_temp*, read from same HDF5 dataset)
        ('mass_bg_TempAboveLim1',           np.float64, s_D),
        ('mass_bg_TempAboveLim2',           np.float64, s_D),
        ('mass_bg_TempAboveLim3',           np.float64, s_D),
        ('mass_bg_out_TempAboveLim1',       np.float64, s_D),
        ('mass_bg_out_TempAboveLim2',       np.float64, s_D),
        ('mass_bg_out_TempAboveLim3',       np.float64, s_D),
        # ── Family E — metals (n_metals × nbins) ──────────────────────────
        ('metals_all',                      np.float64, s_E),
        ('metals_all_emission_weighted',    np.float64, s_E),
        ('metals_all_emission_weighted_e2', np.float64, s_E),
        ('metals_all_out',                  np.float64, s_E),
        ('metals_all_out_fast1',            np.float64, s_E),
        ('metals_all_out_fast2',            np.float64, s_E),
        ('metals_all_out_fast3',            np.float64, s_E),
    ])


# ── PROFI profile loader ─────────────────────────────────────────────────────

# Maps structured-array field name → (HDF5 dataset name, family code)
# Family codes: 'AB' = (n_pt, nbins), 'C' = (pos_dim, n_pt, nbins),
#               'D' = (nbins,), 'E' = (n_met, nbins)
_FIELD_MAP = {
    'np':                              ('np',                              'AB'),
    'np_out':                          ('np_out',                          'AB'),
    'mass':                            ('mass',                            'AB'),
    'vrad_times_mass':                 ('vrad_times_mass',                 'AB'),
    'vLOS_z_times_mass':               ('vLOS_z_times_mass',               'AB'),
    'mass_out':                        ('mass_out',                        'AB'),
    'mass_out_fast1':                  ('mass_out_fast1',                  'AB'),
    'mass_out_fast2':                  ('mass_out_fast2',                  'AB'),
    'mass_out_fast3':                  ('mass_out_fast3',                  'AB'),
    'vrad_times_mass_out':             ('vrad_times_mass_out',             'AB'),
    'vrad_times_mass_out_fast1':       ('vrad_times_mass_out_fast1',       'AB'),
    'vrad_times_mass_out_fast2':       ('vrad_times_mass_out_fast2',       'AB'),
    'vrad_times_mass_out_fast3':       ('vrad_times_mass_out_fast3',       'AB'),
    'vLOS_z_times_mass_out':           ('vLOS_z_times_mass_out',           'AB'),
    'vLOS_times_mass':                 ('vLOS_times_mass',                 'C'),
    'matter_flow_vec':                 ('matter_flow_vec',                 'C'),
    'matter_outflow_vec':              ('matter_outflow_vec',              'C'),
    'Radius':                          ('Radius',                          'D'),
    'temp':                            ('temp',                            'D'),
    'entropy':                         ('entropy',                         'D'),
    'L_bol':                           ('L_bol',                           'D'),
    'L_bol_times_mass':                ('L_bol_times_mass',                'D'),
    'N_e':                             ('N_e',                             'D'),
    'N_H0':                            ('N_H0',                            'D'),
    'mass_temp1':                      ('mass_temp1',                      'D'),
    'mass_temp2':                      ('mass_temp2',                      'D'),
    'mass_temp3':                      ('mass_temp3',                      'D'),
    'mass_out_temp1':                  ('mass_out_temp1',                  'D'),
    'mass_out_temp2':                  ('mass_out_temp2',                  'D'),
    'mass_out_temp3':                  ('mass_out_temp3',                  'D'),
    'temp_out':                        ('temp_out',                        'D'),
    'temp_out_fast1':                  ('temp_out_fast1',                  'D'),
    'temp_out_fast2':                  ('temp_out_fast2',                  'D'),
    'temp_out_fast3':                  ('temp_out_fast3',                  'D'),
    # legacy aliases → same HDF5 datasets as mass_temp*
    'mass_bg_TempAboveLim1':           ('mass_temp1',                      'D'),
    'mass_bg_TempAboveLim2':           ('mass_temp2',                      'D'),
    'mass_bg_TempAboveLim3':           ('mass_temp3',                      'D'),
    'mass_bg_out_TempAboveLim1':       ('mass_out_temp1',                  'D'),
    'mass_bg_out_TempAboveLim2':       ('mass_out_temp2',                  'D'),
    'mass_bg_out_TempAboveLim3':       ('mass_out_temp3',                  'D'),
    'metals_all':                      ('metals_all',                      'E'),
    'metals_all_emission_weighted':    ('metals_all_emission_weighted',    'E'),
    'metals_all_emission_weighted_e2': ('metals_all_emission_weighted_e2', 'E'),
    'metals_all_out':                  ('metals_all_out',                  'E'),
    'metals_all_out_fast1':            ('metals_all_out_fast1',            'E'),
    'metals_all_out_fast2':            ('metals_all_out_fast2',            'E'),
    'metals_all_out_fast3':            ('metals_all_out_fast3',            'E'),
}


def load_profi_profiles(profile_path_snap, sim_label, snapnum,
                        n_partType, nbins_profile, n_metals=10, pos_dim=3):
    """
    Read all PROFI subfiles for one snapshot and return a numpy structured
    array of cumulative profiles.

    Parameters
    ----------
    profile_path_snap : str
        Path to the ``snap<N>`` directory that contains the PROFI .h5 files.
    sim_label : str
        SimulationLabel as set in the PROFI parameter file.  Only the last
        path component is used for filename matching (e.g. if
        SimulationLabel = ``Org/MySim``, the files are named
        ``profile_MySim_snap099_0.h5``).
    snapnum : int
        Snapshot number.
    n_partType : int
        Number of particle types (from the PROFI parameter file or auto-detected).
    nbins_profile : int
        Number of radial bins (from the PROFI parameter file).
    n_metals : int, optional
        Number of metal species (default 10).
    pos_dim : int, optional
        Spatial dimensions (default 3).

    Returns
    -------
    data : np.ndarray (structured)
        Structured array of shape ``(n_halos_total,)`` with cumulative profiles.
    redshift : float
    boxsize : float
        Simulation box size in kpc/h.
    is_hydro_sim : int
        1 for hydro, 0 for DMO.
    """
    # PROFI uses only the last segment of SimulationLabel for file names
    sim_name = os.path.basename(sim_label.rstrip('/'))

    pattern = f"{profile_path_snap}/profile_{sim_name}_snap{snapnum:03d}_*.h5"
    flist = sorted(glob.glob(pattern),
                   key=lambda x: int(x.rsplit('_', 1)[1].split('.')[0]))

    if not flist:
        # Fallback: auto-discover prefix from any subfile 0
        candidates = sorted(glob.glob(
            f"{profile_path_snap}/profile_*_snap{snapnum:03d}_0.h5"))
        if candidates:
            prefix = os.path.basename(candidates[0]).split(f"_snap{snapnum:03d}_")[0]
            pattern = f"{profile_path_snap}/{prefix}_snap{snapnum:03d}_*.h5"
            flist = sorted(glob.glob(pattern),
                           key=lambda x: int(x.rsplit('_', 1)[1].split('.')[0]))

    if not flist:
        raise FileNotFoundError(
            f"No PROFI profile files found matching:\n  {pattern}")

    # Read metadata from the first subfile
    with h5py.File(flist[0], 'r') as f:
        redshift         = float(f['Header'].attrs['Redshift'])
        boxsize          = float(f['Header'].attrs['BoxSize'])
        is_hydro_sim     = int(f['Header'].attrs.get('is_hydro_sim', 1))
        available_fields = set(f['GroupProfile'].keys())

    # Count total halos
    n_halos_total = 0
    for fp in flist:
        with h5py.File(fp, 'r') as f:
            n_halos_total += int(f['Header'].attrs['Ngroups_ThisFile'])

    # Filter field_map to only fields present in the actual HDF5 output
    field_map = {sf: (hf, fam)
                 for sf, (hf, fam) in _FIELD_MAP.items()
                 if hf in available_fields}

    # Build dtype restricted to fields that exist
    full_dtype = build_profi_dtype(n_partType, nbins_profile, n_metals, pos_dim)
    keep_names = [n for n in full_dtype.names if n in field_map]
    dtype = np.dtype([(n, full_dtype[n]) for n in keep_names])

    data = np.zeros(n_halos_total, dtype=dtype)

    idx = 0
    for fp in flist:
        print(f"  Reading {os.path.basename(fp)} …")
        with h5py.File(fp, 'r') as f:
            grp = f['GroupProfile']
            cnt = int(f['Header'].attrs['Ngroups_ThisFile'])
            nxt = idx + cnt
            for sf, (hf, _fam) in field_map.items():
                if sf not in data.dtype.names or hf not in grp:
                    continue
                data[sf][idx:nxt] = grp[hf][()]
            idx = nxt

    return data, redshift, boxsize, is_hydro_sim


# ── Cumulative baryon fraction ────────────────────────────────────────────────

def _cumulative_baryon_fraction(rho_cum):
    """Return the baryonic fraction of the cumulative total density.

    Hydro profile columns follow PROFI's order: gas, DM, stars, BH, DM2.
    The five-type layout includes a second DM component in column 4, which
    must also be excluded from the baryonic mass.
    """
    rho_dm_cum = rho_cum[:, 1, :]
    if rho_cum.shape[1] > 4:
        rho_dm_cum = rho_dm_cum + rho_cum[:, 4, :]
    return 1.0 - rho_dm_cum / np.sum(rho_cum, axis=1)


# ── Cumulative → shell conversion ────────────────────────────────────────────

def make_shell_profiles(profiles_cum_input, rho_mean_arepo, f_b,
                        nbins=None, nbins_temp=None, nbins_vel=None,
                        n_partType=None, n_metals=None, is_hydro_sim=None):
    """
    Convert cumulative (integrated sphere) profiles to shell (differential)
    profiles and compute derived quantities (density, velocity, mass flow, …).

    Parameters
    ----------
    profiles_cum_input : np.ndarray (structured)
        Cumulative profiles as returned by ``load_profi_profiles``.
    rho_mean_arepo : float
        Mean matter density in Arepo/PROFI internal units
        [1e10 Msun h^2 / kpc^3].  Compute as::

            rho_crit_100 = 27.7536627  # [1e10 Msun h^2 / Mpc^3]
            rho_mean_arepo = rho_crit_100 * cosmo_Omega_m / 1e9

    f_b : float
        Cosmic baryon fraction (Omega_b / Omega_m).
    nbins, nbins_temp, nbins_vel, n_partType, n_metals : int, optional
        Auto-detected from the data when None.
        Pass n_partType from the file metadata when no per-particle-type
        fields are present.
    is_hydro_sim : bool or int, optional
        Hydro/DMO flag from the file metadata.  When None, one particle type
        is assumed to mean DMO.  Baryon fractions require a hydro profile
        with a separate DM column.  Other derived quantities are omitted
        when their required cumulative fields are absent.

    Returns
    -------
    profiles : dict
        Dictionary of derived profile arrays.
    profiles_metadata : dict
        Metadata: ``nbins``, ``nbins_temp``, ``nbins_vel``, ``n_partType``,
        ``n_metals``, ``is_hydro_sim``.
    """
    fields    = profiles_cum_input.dtype.names
    num_halos = len(profiles_cum_input)

    # Auto-detect dimensions
    radius_nbins = profiles_cum_input['Radius'].shape[1]
    if nbins is None:
        nbins = radius_nbins
    if nbins < 1 or nbins != radius_nbins:
        raise ValueError('nbins must match the non-empty Radius bin axis.')
    if nbins_temp is None:
        nbins_temp = 1
        while (f'mass_temp{nbins_temp}' in fields
               or f'mass_out_temp{nbins_temp}' in fields):
            nbins_temp += 1
    if nbins_vel is None:
        nbins_vel = 1
        while (f'mass_out_fast{nbins_vel}' in fields
               or f'vrad_times_mass_out_fast{nbins_vel}' in fields):
            nbins_vel += 1
    particle_counts = {
        profiles_cum_input[field].shape[-2]
        for field in fields
        if _FIELD_MAP.get(field, (None, None))[1] in ('AB', 'C')
    }
    if n_partType is None:
        if not particle_counts:
            raise ValueError(
                'Cannot infer n_partType without per-particle-type fields. '
                'Pass n_partType from the profile metadata.')
        n_partType = next(iter(particle_counts))
    if n_partType < 1 or (particle_counts
                         and particle_counts != {n_partType}):
        raise ValueError('n_partType does not match the profile fields.')
    if n_metals is None:
        n_metals = next(
            (profiles_cum_input[field].shape[1] for field in fields
             if _FIELD_MAP.get(field, (None, None))[1] == 'E'), 1)
    if is_hydro_sim is None:
        is_hydro_sim = n_partType > 1

    # Shell values: difference between adjacent cumulative bins
    # (bin 0 = outermost, bin nbins-1 = innermost — PROFI cumulative convention)
    shell = np.zeros(num_halos, dtype=profiles_cum_input.dtype)
    for field in fields:
        shell[field][..., nbins - 1] = profiles_cum_input[field][..., nbins - 1]
        shell[field][..., 0:nbins - 1] = (
            profiles_cum_input[field][..., 0:nbins - 1]
            - profiles_cum_input[field][..., 1:nbins])

    # Radial shell volumes  [kpc/h]^3
    R = profiles_cum_input['Radius']                              # (n_halos, nbins)
    vol_cum  = (4.0 / 3.0) * np.pi * R ** 3
    vol_shell = np.empty_like(vol_cum)
    vol_shell[:, nbins - 1]  = vol_cum[:, nbins - 1]
    vol_shell[:, 0:nbins - 1] = (vol_cum[:, 0:nbins - 1]
                                  - vol_cum[:, 1:nbins])

    profiles = {}
    profiles['Radius']     = R
    profiles['volume']     = vol_shell
    profiles['volume_cum'] = vol_cum

    # ── Density ──────────────────────────────────────────────────────────────
    if 'mass' in fields:
        profiles['rho'] = (shell['mass']
                           / vol_shell[:, None, :]
                           / rho_mean_arepo)                      # rho / rho_mean
        profiles['rho_cum'] = (profiles_cum_input['mass']
                               / vol_cum[:, None, :]
                               / rho_mean_arepo)

        if 'mass_temp1' in fields:
            profiles['rho_tempBinned'] = np.zeros(
                [num_halos, nbins, nbins_temp])
            profiles['rho_tempBinned'][:, :, 0] = (
                (shell['mass'][:, 0, :] - shell['mass_temp1'])
                / vol_shell / rho_mean_arepo)
            for it in range(1, nbins_temp - 1):
                if f'mass_temp{it+1}' in fields:
                    profiles['rho_tempBinned'][:, :, it] = (
                        (shell[f'mass_temp{it}'] - shell[f'mass_temp{it+1}'])
                        / vol_shell / rho_mean_arepo)
            if f'mass_temp{nbins_temp-1}' in fields:
                profiles['rho_tempBinned'][:, :, nbins_temp - 1] = (
                    shell[f'mass_temp{nbins_temp-1}']
                    / vol_shell / rho_mean_arepo)

    # ── Outflow density ───────────────────────────────────────────────────────
    if 'mass_out' in fields:
        profiles['rho_out'] = (shell['mass_out']
                               / vol_shell[:, None, :]
                               / rho_mean_arepo)

        if 'mass_out_temp1' in fields:
            profiles['rho_out_tempBinned'] = np.zeros(
                [num_halos, nbins, nbins_temp])
            profiles['rho_out_tempBinned'][:, :, 0] = (
                (shell['mass_out'][:, 0, :] - shell['mass_out_temp1'])
                / vol_shell / rho_mean_arepo)
            for it in range(1, nbins_temp - 1):
                if f'mass_out_temp{it+1}' in fields:
                    profiles['rho_out_tempBinned'][:, :, it] = (
                        (shell[f'mass_out_temp{it}']
                         - shell[f'mass_out_temp{it+1}'])
                        / vol_shell / rho_mean_arepo)
            if f'mass_out_temp{nbins_temp-1}' in fields:
                profiles['rho_out_tempBinned'][:, :, nbins_temp - 1] = (
                    shell[f'mass_out_temp{nbins_temp-1}']
                    / vol_shell / rho_mean_arepo)

        if 'mass_out_fast1' in fields:
            profiles['rho_out_velBinned'] = np.zeros(
                [num_halos, n_partType, nbins, nbins_vel])
            profiles['rho_out_velBinned'][:, :, :, 0] = (
                (shell['mass_out'] - shell['mass_out_fast1'])
                / vol_shell[:, None, :] / rho_mean_arepo)
            for iv in range(1, nbins_vel - 1):
                if f'mass_out_fast{iv+1}' in fields:
                    profiles['rho_out_velBinned'][:, :, :, iv] = (
                        (shell[f'mass_out_fast{iv}']
                         - shell[f'mass_out_fast{iv+1}'])
                        / vol_shell[:, None, :] / rho_mean_arepo)
            if f'mass_out_fast{nbins_vel-1}' in fields:
                profiles['rho_out_velBinned'][:, :, :, nbins_vel - 1] = (
                    shell[f'mass_out_fast{nbins_vel-1}']
                    / vol_shell[:, None, :] / rho_mean_arepo)

        if 'mass' in fields:
            profiles['rho_in'] = (
                (shell['mass'] - shell['mass_out'])
                / vol_shell[:, None, :] / rho_mean_arepo)

    # ── Radial velocity & mass flow ───────────────────────────────────────────
    if 'vrad_times_mass' in fields:
        if 'mass' in fields:
            profiles['v_r'] = shell['vrad_times_mass'] / shell['mass']
        profiles['mass_flow'] = (
            shell['vrad_times_mass']
            / (R[:, None, :] * _KPC_TO_KM))

        if 'matter_flow_vec' in fields:
            profiles['matter_flow_vec'] = (
                shell['matter_flow_vec']
                / shell['vrad_times_mass'][:, None, :])

    if 'vrad_times_mass_out' in fields:
        if 'mass_out' in fields:
            profiles['v_r_out'] = (shell['vrad_times_mass_out']
                                   / shell['mass_out'])
        profiles['mass_outflow'] = (
            shell['vrad_times_mass_out']
            / (R[:, None, :] * _KPC_TO_KM))
        if 'vrad_times_mass' in fields:
            if 'mass' in fields and 'mass_out' in fields:
                profiles['v_r_in'] = (
                    (shell['vrad_times_mass'] - shell['vrad_times_mass_out'])
                    / (shell['mass'] - shell['mass_out']))
            profiles['mass_inflow'] = (
                (shell['vrad_times_mass'] - shell['vrad_times_mass_out'])
                / (R[:, None, :] * _KPC_TO_KM))

        if 'vrad_times_mass_out_fast1' in fields:
            profiles['mass_outflow_velBinned'] = np.zeros(
                [num_halos, n_partType, nbins, nbins_vel])
            profiles['mass_outflow_velBinned'][:, :, :, 0] = (
                (shell['vrad_times_mass_out']
                 - shell['vrad_times_mass_out_fast1'])
                / (R[:, None, :] * _KPC_TO_KM))
            for iv in range(1, nbins_vel - 1):
                if f'vrad_times_mass_out_fast{iv+1}' in fields:
                    profiles['mass_outflow_velBinned'][:, :, :, iv] = (
                        (shell[f'vrad_times_mass_out_fast{iv}']
                         - shell[f'vrad_times_mass_out_fast{iv+1}'])
                        / (R[:, None, :] * _KPC_TO_KM))
            if f'vrad_times_mass_out_fast{nbins_vel-1}' in fields:
                profiles['mass_outflow_velBinned'][:, :, :, nbins_vel - 1] = (
                    shell[f'vrad_times_mass_out_fast{nbins_vel-1}']
                    / (R[:, None, :] * _KPC_TO_KM))

        if 'matter_outflow_vec' in fields:
            profiles['matter_outflow_vec'] = (
                shell['matter_outflow_vec']
                / shell['vrad_times_mass_out'][:, None, :])
            if ('matter_flow_vec' in fields
                    and 'vrad_times_mass' in fields):
                profiles['matter_inflow_vec'] = (
                    (shell['matter_flow_vec'] - shell['matter_outflow_vec'])
                    / (shell['vrad_times_mass'][:, None, :]
                       - shell['vrad_times_mass_out'][:, None, :]))

    # ── Line-of-sight velocity ────────────────────────────────────────────────
    if 'vLOS_z_times_mass' in fields and 'mass' in fields:
        profiles['vLOS_z'] = (shell['vLOS_z_times_mass']
                              / shell['mass'])
    if 'vLOS_z_times_mass_out' in fields and 'mass_out' in fields:
        profiles['vLOS_z_out'] = (shell['vLOS_z_times_mass_out']
                                  / shell['mass_out'])
        if 'vLOS_z_times_mass' in fields and 'mass' in fields:
            profiles['vLOS_z_in'] = (
                (shell['vLOS_z_times_mass'] - shell['vLOS_z_times_mass_out'])
                / (shell['mass'] - shell['mass_out']))

    # ── Temperature ──────────────────────────────────────────────────────────
    if 'temp' in fields and 'mass' in fields:
        profiles['temp'] = shell['temp'] / shell['mass'][:, 0, :]
    if 'temp_out' in fields and 'mass_out' in fields:
        profiles['temp_out'] = (shell['temp_out']
                                / shell['mass_out'][:, 0, :])
        if 'temp' in fields and 'mass' in fields:
            profiles['temp_in'] = (
                (shell['temp'] - shell['temp_out'])
                / (shell['mass'][:, 0, :] - shell['mass_out'][:, 0, :]))
        if 'temp_out_fast1' in fields and 'mass_out_fast1' in fields:
            profiles['temp_out_velBinned'] = np.zeros(
                [num_halos, nbins, nbins_vel])
            profiles['temp_out_velBinned'][:, :, 0] = (
                (shell['temp_out'] - shell['temp_out_fast1'])
                / (shell['mass_out'][:, 0, :]
                   - shell['mass_out_fast1'][:, 0, :]))
            for iv in range(1, nbins_vel - 1):
                if all(name in fields for name in (
                        f'temp_out_fast{iv}', f'temp_out_fast{iv+1}',
                        f'mass_out_fast{iv}', f'mass_out_fast{iv+1}')):
                    profiles['temp_out_velBinned'][:, :, iv] = (
                        (shell[f'temp_out_fast{iv}']
                         - shell[f'temp_out_fast{iv+1}'])
                        / (shell[f'mass_out_fast{iv}'][:, 0, :]
                           - shell[f'mass_out_fast{iv+1}'][:, 0, :]))
            if (f'temp_out_fast{nbins_vel-1}' in fields
                    and f'mass_out_fast{nbins_vel-1}' in fields):
                profiles['temp_out_velBinned'][:, :, nbins_vel - 1] = (
                    shell[f'temp_out_fast{nbins_vel-1}']
                    / shell[f'mass_out_fast{nbins_vel-1}'][:, 0, :])

    # ── Metallicity ───────────────────────────────────────────────────────────
    if 'metals_all' in fields and 'mass' in fields:
        profiles['metals_all'] = (shell['metals_all']
                                  / shell['mass'][:, None, 0, :])
        profiles['metallicity'] = np.sum(
            profiles['metals_all'][:, 2:n_metals, :], axis=1)
    if 'metals_all_out' in fields and 'mass_out' in fields:
        profiles['metals_all_out'] = (shell['metals_all_out']
                                      / shell['mass_out'][:, None, 0, :])
        profiles['metallicity_out'] = np.sum(
            profiles['metals_all_out'][:, 2:n_metals, :], axis=1)
        if 'metals_all' in fields and 'mass' in fields:
            profiles['metals_all_in'] = (
                (shell['metals_all'] - shell['metals_all_out'])
                / (shell['mass'][:, None, 0, :]
                   - shell['mass_out'][:, None, 0, :]))
            profiles['metallicity_in'] = np.sum(
                profiles['metals_all_in'][:, 2:n_metals, :], axis=1)

        if ('metals_all_out_fast1' in fields
                and 'mass_out_fast1' in fields):
            profiles['metals_all_out_velBinned'] = np.zeros(
                [num_halos, n_metals, nbins, nbins_vel])
            profiles['metals_all_out_velBinned'][:, :, :, 0] = (
                (shell['metals_all_out'] - shell['metals_all_out_fast1'])
                / (shell['mass_out'][:, None, 0, :]
                   - shell['mass_out_fast1'][:, None, 0, :]))
            for iv in range(1, nbins_vel - 1):
                if all(name in fields for name in (
                        f'metals_all_out_fast{iv}',
                        f'metals_all_out_fast{iv+1}',
                        f'mass_out_fast{iv}', f'mass_out_fast{iv+1}')):
                    profiles['metals_all_out_velBinned'][:, :, :, iv] = (
                        (shell[f'metals_all_out_fast{iv}']
                         - shell[f'metals_all_out_fast{iv+1}'])
                        / (shell[f'mass_out_fast{iv}'][:, None, 0, :]
                           - shell[f'mass_out_fast{iv+1}'][:, None, 0, :]))
            if (f'metals_all_out_fast{nbins_vel-1}' in fields
                    and f'mass_out_fast{nbins_vel-1}' in fields):
                profiles['metals_all_out_velBinned'][:, :, :, nbins_vel - 1] = (
                    shell[f'metals_all_out_fast{nbins_vel-1}']
                    / shell[f'mass_out_fast{nbins_vel-1}'][:, None, 0, :])
            profiles['metallicity_out_velBinned'] = np.sum(
                profiles['metals_all_out_velBinned'][:, 2:n_metals, :, :],
                axis=1)

    # ── Baryon fraction ───────────────────────────────────────────────────────
    if is_hydro_sim and n_partType >= 2 and 'rho_cum' in profiles:
        profiles['baryon_fraction_cum'] = (
            _cumulative_baryon_fraction(profiles['rho_cum'])
        ) / f_b                                                    # normalised by cosmic f_b

    # Cast floats to float32, ints to int32 to save memory
    for key in profiles:
        arr = profiles[key]
        if arr.dtype == np.float64:
            profiles[key] = arr.astype(np.float32)
        elif arr.dtype == np.int64:
            profiles[key] = arr.astype(np.int32)

    profiles_metadata = {
        'nbins':      nbins,
        'nbins_temp': nbins_temp,
        'nbins_vel':  nbins_vel,
        'n_partType': n_partType,
        'n_metals':   n_metals,
        'is_hydro_sim': int(is_hydro_sim),
    }
    return profiles, profiles_metadata


# ── HDF5 I/O utilities ───────────────────────────────────────────────────────

def save_dict_to_hdf5(h5group, dic, compression=4):
    """
    Recursively save a dictionary to an HDF5 group.

    Parameters
    ----------
    h5group : h5py.Group or h5py.File
    dic : dict
        Values can be numpy arrays, scalars, or nested dicts.
    compression : int, optional
        Gzip compression level 0-9 (default 4).
    """
    for key, item in dic.items():
        if isinstance(item, dict):
            sub = h5group.create_group(key)
            save_dict_to_hdf5(sub, item, compression=compression)
        else:
            h5group.create_dataset(
                key, data=item,
                compression='gzip', compression_opts=compression,
                shuffle=True, chunks=True)


def add_header_attrs(h5file, keys, values):
    """
    Write key-value pairs as attributes of the ``/Header`` group.

    Parameters
    ----------
    h5file : h5py.File
    keys : list of str
    values : list
    """
    if 'Header' not in h5file:
        grp = h5file.create_group('Header')
    else:
        grp = h5file['Header']
    for k, v in zip(keys, values):
        grp.attrs[k] = v


# ── Optional memory reporter ─────────────────────────────────────────────────

def print_memory_mb():
    """Print current RSS memory usage in MB (requires psutil)."""
    try:
        import psutil, os as _os
        usage = psutil.Process(_os.getpid()).memory_info().rss / 1024**2
        print(f"  Memory: {usage:.1f} MB")
        return usage
    except ImportError:
        return None


# ── HDF5 reader ──────────────────────────────────────────────────────────────

def open_hdf5(file_path, fields=None, print_header=False):
    """
    Read selected (or all) top-level datasets from an HDF5 file.

    Parameters
    ----------
    file_path : str
    fields : list of str, optional
        If None, all top-level datasets are loaded.
    print_header : bool, optional
        If True, print the attributes of the Header group.

    Returns
    -------
    data : dict
        {field_name: numpy_array}
    """
    data = {}
    with h5py.File(file_path, 'r') as f:
        if print_header and 'Header' in f:
            print('Header attributes:')
            for k, v in f['Header'].attrs.items():
                print(f'  {k}: {v}')
        keys = fields if fields is not None else list(f.keys())
        for k in keys:
            if k not in f:
                continue
            item = f[k]
            if isinstance(item, h5py.Dataset):
                data[k] = item[()]
            # skip groups silently
    return data


# ── Histogram utility ─────────────────────────────────────────────────────────

def get_bin_centers(edges):
    """Return the centre of each bin given an array of bin edges."""
    edges = np.asarray(edges)
    return (edges[:-1] + edges[1:]) * 0.5


# ── Closure radius ────────────────────────────────────────────────────────────

def calculate_closure_radius(profiles, halo,
                              f_b,
                              r_cl_normalized_minlim=0.3,
                              r_cl_normalized_maxlim=None,
                              error_fb_planck_percent=0.05):
    """
    Calculate the closure radius for each halo.

    The closure radius is the innermost radial bin at or beyond the minimum
    radius where the cumulative baryon fraction lies within
    ``(1 ± error_fb_planck_percent) * f_b``. If no bin meets both conditions,
    return the fallback radius supplied by ``r_cl_normalized_maxlim``.

    Parameters
    ----------
    profiles : dict
        Shell profiles dict as returned by ``make_shell_profiles``.
        Must contain ``'Radius'`` and ``'rho_cum'``.
    halo : dict
        Halo catalogue with at least ``'Group_R_Crit200'``.
    f_b : float
        Cosmic baryon fraction (Omega_b / Omega_m).
    r_cl_normalized_minlim : float, optional
        Minimum closure radius in units of R200c (default 0.3).
    r_cl_normalized_maxlim : float, array_like or None, optional
        Fallback radius in units of R200c, used only when no eligible closure
        bin is found. A scalar applies to all halos; an array supplies a
        fallback for each halo. None uses each profile's outer edge.
        Eligible closure bins take precedence over this fallback.
    error_fb_planck_percent : float, optional
        Fractional tolerance around f_b (default 0.05 = 5%).

    Returns
    -------
    output : dict
        ``'Rc'``      — closure radius in kpc/h (float32 array, n_halos)
        ``'Rc_norm'`` — closure radius in units of R200c (float32 array, n_halos)
    """
    n_halos = profiles['Radius'].shape[0]
    nbins   = profiles['Radius'].shape[1]

    radius_norm = profiles['Radius'] / halo['Group_R_Crit200'][:, None]

    rho_baryon_cum = _cumulative_baryon_fraction(profiles['rho_cum'])

    if r_cl_normalized_maxlim is None:
        maxlim = np.max(radius_norm, axis=1)
    elif np.isscalar(r_cl_normalized_maxlim):
        maxlim = np.full(n_halos, r_cl_normalized_maxlim, dtype=np.float32)
    else:
        maxlim = np.asarray(r_cl_normalized_maxlim, dtype=np.float32)

    f_b_lo = f_b * (1.0 - error_fb_planck_percent)
    f_b_hi = f_b * (1.0 + error_fb_planck_percent)

    # For each halo: last bin where baryon fraction is in [f_b_lo, f_b_hi]
    # and normalised radius is above the minimum limit.
    mask = ((rho_baryon_cum >= f_b_lo)
            & (rho_baryon_cum <= f_b_hi)
            & (radius_norm >= r_cl_normalized_minlim))
    valid = mask.any(axis=1)
    last_idx = nbins - 1 - np.argmax(mask[:, ::-1], axis=1)

    Rc_norm = np.where(
        valid,
        radius_norm[np.arange(n_halos), last_idx],
        maxlim).astype(np.float32)

    output = {
        'Rc_norm': Rc_norm,
        'Rc':      (Rc_norm * halo['Group_R_Crit200']).astype(np.float32),
    }
    return output


def calculate_baryon_fraction_at_radius(profiles, halo, f_b, radius_in_R200c=1.0):
    """
    Calculate the cumulative baryon fraction (normalised by f_b) at a given radius.

    Parameters
    ----------
    profiles : dict
        Shell profiles dict containing ``'Radius'`` and ``'rho_cum'``.
    halo : dict
        Halo catalogue with at least ``'Group_R_Crit200'``.
    f_b : float
        Cosmic baryon fraction (Omega_b / Omega_m).
    radius_in_R200c : float, optional
        Radius at which to evaluate the baryon fraction, in units of R200c
        (default 1.0 = R200c).

    Returns
    -------
    baryon_fraction : np.ndarray (float32, n_halos)
        Baryon fraction / f_b at the requested radius.
    """
    n_halos = profiles['Radius'].shape[0]
    radius_norm = profiles['Radius'] / halo['Group_R_Crit200'][:, None]

    rho_baryon_cum = _cumulative_baryon_fraction(profiles['rho_cum'])

    radius_idx = np.argmin(np.abs(radius_norm - radius_in_R200c), axis=1)
    baryon_fraction = (
        rho_baryon_cum[np.arange(n_halos), radius_idx] / f_b
    ).astype(np.float32)
    return baryon_fraction


# ── Compensation radius ───────────────────────────────────────────────────────

def calculate_compensation_radius(profiles_hydro, profiles_dmo,
                                  halo_hydro, halo_dmo,
                                  rho_mean_arepo,
                                  hydro_idx=None, dmo_idx=None,
                                  r_comp_normalized_minlim=0.5,
                                  error_r=0.0093):
    """
    Calculate the compensation radius for matched hydro/DMO halo pairs.

    The compensation radius ``R_comp`` is the halocentric radius at which the
    cumulative total mass in the hydro run equals that in its DMO counterpart.
    Starting from the outermost bin inside R200c where the ratio
    ``M_hydro / M_DMO`` leaves the band ``[1 - error_r, 1 + error_r]``, ``R_comp``
    is the first radius further out where the ratio re-enters the band (linearly
    interpolated at the crossing).  If the ratio never leaves the band between
    the minimum search radius and R200c, ``R_comp`` is the innermost searched
    radius; if it never re-enters the band, ``R_comp`` is the outermost searched
    radius.

    Parameters
    ----------
    profiles_hydro, profiles_dmo : dict
        Shell-profile dicts (as returned by ``make_shell_profiles``) for the
        hydro run and its DMO counterpart.  Must contain ``'Radius'``,
        ``'rho_cum'`` and ``'volume_cum'``.
    halo_hydro, halo_dmo : dict
        Halo catalogues with at least ``'Group_M_Crit200'`` and
        ``'Group_R_Crit200'``, aligned row-by-row with the profile arrays.
    rho_mean_arepo : float
        Mean matter density in PROFI internal units [1e10 Msun h^2 / kpc^3].
    hydro_idx, dmo_idx : array of int, optional
        Indices of the matched (hydro, DMO) profile rows.  Defaults to
        index-matching over the shared rows.
    r_comp_normalized_minlim : float, optional
        Minimum search radius in units of R200c (default 0.5).
    error_r : float, optional
        Fractional tolerance on ``M_hydro / M_DMO`` (default 0.0093).

    Returns
    -------
    output : dict
        Hydro-indexed arrays (length ``n_hydro``); hydro rows without a match
        are left as ``NaN``.

        ``'R_comp'``              — compensation radius in kpc/h (float32)
        ``'R_comp_norm'``         — R_comp / R200c_hydro (float32)
        ``'R_comp_norm_DMO'``     — R_comp / R200c_DMO   (float32)
        ``'Group_M_Crit200'``     — hydro halo mass  [1e10 Msun/h] (float32)
        ``'Group_R_Crit200'``     — hydro halo radius [kpc/h] (float32)
        ``'Group_M_Crit200_DMO'`` — matched DMO halo mass  [1e10 Msun/h] (float32)
        ``'Group_R_Crit200_DMO'`` — matched DMO halo radius [kpc/h] (float32)
    """

    n_hydro = profiles_hydro['Radius'].shape[0]
    n_dmo   = profiles_dmo['Radius'].shape[0]

    if hydro_idx is None or dmo_idx is None:
        n_pairs   = min(n_hydro, n_dmo,
                        len(halo_hydro['Group_R_Crit200']),
                        len(halo_dmo['Group_R_Crit200']))
        hydro_idx = np.arange(n_pairs)
        dmo_idx   = np.arange(n_pairs)
    hydro_idx = np.asarray(hydro_idx, dtype=np.int64)
    dmo_idx   = np.asarray(dmo_idx,   dtype=np.int64)

    R200c_h = halo_hydro['Group_R_Crit200']
    R200c_d = halo_dmo['Group_R_Crit200']
    M200c_h = halo_hydro['Group_M_Crit200']
    M200c_d = halo_dmo['Group_M_Crit200']

    # Cumulative enclosed mass: sum over particle types, times the cumulative
    # shell volume, times the mean matter density.  Shape (n_halos, nbins).
    M_h_all = (np.sum(profiles_hydro['rho_cum'], axis=1)
               * profiles_hydro['volume_cum'] * rho_mean_arepo)
    M_d_all = (np.sum(profiles_dmo['rho_cum'], axis=1)
               * profiles_dmo['volume_cum'] * rho_mean_arepo)

    R_comp          = np.full(n_hydro, np.nan, dtype=np.float32)
    R_comp_norm     = np.full(n_hydro, np.nan, dtype=np.float32)
    R_comp_norm_DMO = np.full(n_hydro, np.nan, dtype=np.float32)
    M200c_out       = np.full(n_hydro, np.nan, dtype=np.float32)
    R200c_out       = np.full(n_hydro, np.nan, dtype=np.float32)
    M200c_d_out     = np.full(n_hydro, np.nan, dtype=np.float32)
    R200c_d_out     = np.full(n_hydro, np.nan, dtype=np.float32)

    for i_h, i_d in zip(hydro_idx, dmo_idx):
        r_h = profiles_hydro['Radius'][i_h]
        r_d = profiles_dmo['Radius'][i_d]

        # Interpolate the DMO enclosed mass onto the hydro radial grid.
        M_d_i = np.interp(r_h, r_d[::-1], M_d_all[i_d][::-1],
                          left=0.0, right=M_d_all[i_d][0])

        R200c = float(R200c_h[i_h])
        vmask = (M_d_i > 0) & (r_h / R200c >= r_comp_normalized_minlim)
        if not np.any(vmask):
            continue

        vidx  = np.where(vmask)[0]
        r_v   = r_h[vidx]
        ratio = M_h_all[i_h][vidx] / M_d_i[vidx]

        # Outermost violation inside R200c, then the first radius further out
        # where the ratio re-enters the tolerance band.
        viol    = np.abs(ratio - 1.0) > error_r
        viol_in = np.where(viol & (r_v <= R200c))[0]
        if len(viol_in) == 0:
            r_comp = r_v[-1]
        else:
            jo   = viol_in[0]
            conv = np.where(~viol[:jo])[0]
            if len(conv) == 0:
                r_comp = r_v[0]
            else:
                k = conv[-1]
                r1, r2     = r_v[k], r_v[k + 1]
                rat1, rat2 = ratio[k], ratio[k + 1]
                tgt  = 1.0 + error_r if rat2 > 1.0 + error_r else 1.0 - error_r
                r_comp = (r1 + (tgt - rat1) / (rat2 - rat1) * (r2 - r1)
                          if rat1 != rat2 else r2)

        R_comp[i_h]          = r_comp
        R_comp_norm[i_h]     = r_comp / R200c
        R_comp_norm_DMO[i_h] = r_comp / R200c_d[i_d]
        M200c_out[i_h]       = M200c_h[i_h]
        R200c_out[i_h]       = R200c
        M200c_d_out[i_h]     = M200c_d[i_d]
        R200c_d_out[i_h]     = R200c_d[i_d]

    output = {
        'R_comp':              R_comp,
        'R_comp_norm':         R_comp_norm,
        'R_comp_norm_DMO':     R_comp_norm_DMO,
        'Group_M_Crit200':     M200c_out,
        'Group_R_Crit200':     R200c_out,
        'Group_M_Crit200_DMO': M200c_d_out,
        'Group_R_Crit200_DMO': R200c_d_out,
    }
    return output
