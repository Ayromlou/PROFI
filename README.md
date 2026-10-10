# PROFI — Radial Profile Code

**Author:** M. Reza Ayromlou (ayromlou@uni-bonn.de; ayromlou@gmail.com)

PROFI computes radial profiles of density, velocity, temperature, and metallicity for dark-matter halos in cosmological simulations. It reads particle data and friends-of-friends (FoF) catalogues in HDF5 format and supports both hydrodynamic (gas, dark matter, stars, and black holes) and dark-matter-only (DMO) runs.

PROFI measures two characteristic radii for each halo: the **Closure Radius**, the halocentric distance within which all baryons associated with the halo are contained ([Ayromlou et al. 2023b](https://ui.adsabs.harvard.edu/abs/2023MNRAS.524.5391A/abstract)), and the **Compensation Radius**, the halocentric distance at which the cumulative total mass of a halo in a hydrodynamic simulation equals its gravity-only counterpart ([Ayromlou et al. 2026](https://ui.adsabs.harvard.edu/abs/2026arXiv261007140A/abstract)).

The native input layout is the **AREPO-style HDF5 format** (e.g. Illustris, TNG, iCluster). With minor adjustments to the input, other simulation codes, including **GADGET4**, **GIZMO**, and **SWIFT**, can be easily accomodated. A conversion script is available from the author on request, and will be made publicly available in the near future.

If you use the code, please cite ([Ayromlou et al. 2026](https://ui.adsabs.harvard.edu/abs/2026arXiv261007140A/abstract)).

---

## Workflow

The pipeline consists of three sequential steps, followed by an optional fourth step for hydro/DMO comparisons.

```
Step 1 — PROFI (C, compiled)
  Reads particle data and FoF catalogues.
  Outputs one cumulative radial profile file per FoF subfile, per snapshot.

    → <OutputDir>/snap<N>/profile_<label>_snap<NNN>_<K>.h5

Step 2 — Postprocessing/process_profiles_to_shells.py  (Python)
  Merges per-subfile profiles into a single array per snapshot.
  Converts cumulative (sphere-integrated) profiles to shell (differential) profiles.
  Computes density, velocity, temperature, and metallicity profiles.

    → <OutputDir>/snap<N>/Group_Profiles_<label>_snap<N>_Processed.hdf5

Step 3 — Postprocessing/calculate_closure_radius.py  (Python)
  Reads processed profiles and the halo catalogue.
  Computes the closure radius and baryon fraction for each halo.
  Intended for hydrodynamic simulations; it is not applicable to DMO runs.
  Requires a catalogue reader adapted to your simulation format.
  If no bin meets the closure conditions, r_cl_normalized_maxlim supplies a
  fallback radius (None uses each profile's outer edge). Eligible closure bins
  take precedence over this fallback.

    → <OutputDir>/snap<N>/Group_Closure_Radius_<label>_snap<N>.hdf5

Step 4 (optional) — Postprocessing/calculate_compensation_radius.py  (Python)
  Reads the processed profiles of a hydrodynamic run and its dark-matter-only
  (DMO) counterpart, and computes the compensation radius (the radius where
  the cumulative total mass in the hydro run equals the cumulative DMO mass)
  for matched pairs.
  Requires both a hydro and a DMO profile set, and assumes the two are
  index-matched (row i ↔ row i); if they are not, perform the hydro/DMO matching
  as a preprocessing step so the arrays are aligned before running this script.
  If the mass ratio stays within tolerance from the minimum search radius to
  R200c, the result is the innermost searched radius. If it leaves the band and
  never re-enters farther out, the result is the outermost searched radius.

    → <OutputDir>/snap<N>/Group_Compensation_Radius_<label>_snap<N>.hdf5
```

**Python dependencies:** Steps 2–4 require `numpy` and `h5py` as their only third-party packages. They also use the shared local helpers in `Postprocessing/profi_helpers.py` and standard-library modules.

---

## Requirements

### Step 1 — PROFI (C code)


| Dependency                   | Notes                                      |
| ------------------------------ | -------------------------------------------- |
| C compiler (gcc ≥ 9 or icc) | with OpenMP support                        |
| HDF5 library                 | `h5pcc` wrapper (comes with parallel HDF5) |

The Makefile uses `h5pcc` as the compiler wrapper; it automatically links HDF5 and zlib.  If your installation uses a different wrapper (e.g. `h5cc`), edit the `CC` line in the Makefile.

The default build is portable across x86-64 hosts.  For maximum performance on the build machine you may enable CPU-specific tuning with `make ARCH="-march=native -mtune=native"`.

### Steps 2–4 — Python scripts


| Package | Version    |
| --------- | ------------ |
| Python  | ≥ 3.8     |
| numpy   | any recent |
| h5py    | any recent |

---

## Quick Start

```bash
# ── Step 1: build ─────────────────────────────────────────────────────────────
make

# Copy and fill in the template parameter file
cp params/params_template.txt params/params_MySim.txt
$EDITOR params/params_MySim.txt

# Run one subfile (snapshot 99, subfile 0)
export OMP_NUM_THREADS=8
./PROFI params/params_MySim.txt 99 99 0 0

# For a snapshot with subfiles 0-64, run one process per subfile. Adjust the
# range and thread count to match your snapshot and available CPU allocation.
pids=()
for sf in {0..64}; do
    OMP_NUM_THREADS=2 ./PROFI params/params_MySim.txt 99 99 "$sf" "$sf" &
    pids[$sf]=$!
done
failed=0
for sf in "${!pids[@]}"; do
    if wait "${pids[$sf]}"; then
        :
    else
        worker_status=$?
        printf 'ERROR: subfile %s exited with status %s.\n' \
            "$sf" "$worker_status" >&2
        failed=1
    fi
done
if (( failed )); then
    exit 1
fi

# ── Step 2: convert cumulative → shell profiles ───────────────────────────────
# Edit the Configuration block at the top of the script, then:
python Postprocessing/process_profiles_to_shells.py

# ── Step 3: compute closure radii ─────────────────────────────────────────────
# Implement load_halo_catalogue() in the script, edit settings, then:
python Postprocessing/calculate_closure_radius.py

# ── Step 4 (optional): compute compensation radii (hydro vs. DMO) ─────────────
# Implement load_halo_catalogue() in the script, edit settings, then:
python Postprocessing/calculate_compensation_radius.py
```

---

## PROFI Command-Line Syntax

```
./PROFI <parameterfile> <snapnum_i> <snapnum_f> <subfilenr_i> <subfilenr_f> [part_in_profile_method]
```


| Argument                      | Description                                                                                                                                      |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------- |
| `parameterfile`               | Path to the PROFI parameter file                                                                                                                 |
| `snapnum_i` / `snapnum_f`     | First / last snapshot to process                                                                                                                 |
| `subfilenr_i` / `subfilenr_f` | First / last FoF subfile to process                                                                                                              |
| `part_in_profile_method`      | *(optional)* `0` = standard particle selection (default); `3` = unbound particles. Modes `1` and `2` are not implemented in this public version. |

---

## Parameter File

See [`params/params_template.txt`](params/params_template.txt) for a fully documented template.

**Required parameters:**


| Tag                            | Type   | Description                                                             |
| -------------------------------- | -------- | ------------------------------------------------------------------------- |
| `OutputDir`                    | string | Directory for PROFI output files                                        |
| `SimulationDir`                | string | Root directory of the simulation                                        |
| `InputDir`                     | string | Root directory for auxiliary input files                                |
| `SimulationLabel`              | string | Short label used in output file names                                   |
| `NumberOfSnapshots`            | int    | Total number of snapshots                                               |
| `BinSize`                      | double | Spatial grid cell size (kpc/h)                                          |
| `M200Min` / `M200Max`          | double | Halo mass cuts (10¹⁰ M☉)                                             |
| `nbins_profile`                | int    | Number of radial bins                                                   |
| `nRvir_profile`                | int    | Profile outer radius in units of R₂₀₀                                |
| `Temp_lim1/2/3`                | double | Gas temperature thresholds (K)                                          |
| `temp_starForming_gas_cell`    | double | Effective temperature for star-forming gas cells (K)                    |
| `v_rad_lim1/2/3`               | double | Radial velocity thresholds (km/s)                                       |
| `n_metals`                     | int    | Number of metal species in`GFM_Metals` (10 for TNG/EAGLE, 11 for SIMBA) |
| `projection_flag`              | int    | 0 = 3-D spherical, 1 = 2-D projected                                    |
| `is_hydro_sim`                 | int    | 1 = hydro, 0 = DMO                                                      |
| `calculate_all_gas_properties` | int    | 1 = compute temperature/X-ray/metallicity, 0 = skip                     |

**Optional parameters** (defaults shown):


| Tag                      | Default | Description                                                                         |
| -------------------------- | --------- | ------------------------------------------------------------------------------------- |
| `n_partType`             | 3       | Particle types: 1 = DM, 2 = gas+DM, 3 = gas+DM+stars, 5 = +BH+DM2                   |
| `cosmo_h`                | 0.6774  | Hubble parameter (Planck 2015)                                                      |
| `cosmo_Omega_m`          | 0.3089  | Total matter density                                                                |
| `cosmo_Omega_b`          | 0.0486  | Baryon density                                                                      |
| `cosmo_Omega_Lambda`     | 0.6911  | Dark energy density                                                                 |
| `output_*` flags         | 1       | Granular output control; see`params/params_template.txt` for the full list          |
| `part_in_profile_method` | 0       | Particle selection mode (see command-line table; modes 1 and 2 are not implemented) |
| `cluster_subfile_mode`   | 0       | Set to 1 only for TNG-Cluster-like catalogues                                       |

---

## Output Files

**Step 1 — PROFI** writes one HDF5 file per FoF subfile:

```
<OutputDir>/snap<N>/profile_<SimulationLabel>_snap<NNN>_<K>.h5
```

Scalar profiles have shape `[N_halos, nbins]`. Per-particle-type profiles have shape `[N_halos, n_partType, nbins]`; vector profiles also include a coordinate axis, and metal profiles include a metal-species axis.

**Step 2** merges and post-processes these into one file per snapshot:

```
<OutputDir>/snap<N>/Group_Profiles_<SimulationLabel>_snap<N>_Processed.hdf5
```

**Step 3** writes one closure-radius file per snapshot:

```
<OutputDir>/snap<N>/Group_Closure_Radius_<SimulationLabel>_snap<N>.hdf5
```

**Step 4** (optional, hydro/DMO) writes one compensation-radius file per snapshot:

```
<OutputDir>/snap<N>/Group_Compensation_Radius_<SimulationLabel>_snap<N>.hdf5
```

---

## Expected HDF5 Data Layout

PROFI reads the following paths from each snapshot and FoF catalogue. The C code checks for optional datasets before reading them. Missing optional values are generally zero-filled; if `GFM_Metals` is absent, the related profile arrays remain zero and PROFI prints a warning.

**FoF catalogue** (`output/groups_NNN/fof_subhalo_tab_NNN.K.hdf5`):


| Dataset                  | Description            |
| -------------------------- | ------------------------ |
| `/Group/GroupPos`        | Halo positions         |
| `/Group/GroupVel`        | Halo velocities        |
| `/Group/Group_R_Crit200` | R₂₀₀ (kpc/h)        |
| `/Group/Group_M_Crit200` | M₂₀₀ (10¹⁰ M☉/h) |

**Particle snapshot** (`output/snapdir_NNN/snap_NNN.K.hdf5`):


| Dataset                                            | Description                                                   |
| ---------------------------------------------------- | --------------------------------------------------------------- |
| `/Header` (group)                                  | Attributes: BoxSize, MassTable, NumFilesPerSnapshot, Redshift |
| `/PartTypeX/Coordinates`                           | Positions (float64)                                           |
| `/PartTypeX/Velocities`                            | Peculiar velocities (float32)                                 |
| `/PartTypeX/Masses`                                | Particle masses (not needed for DM if MassTable[1] > 0)       |
| `/PartType0/InternalEnergy` or `InternalEnergyOld` | *(optional)*                                                  |
| `/PartType0/Density`                               | Required for hydro runs; used by the gas/X-ray calculations   |
| `/PartType0/NeutralHydrogenAbundance`              | *(optional)*                                                  |
| `/PartType0/ElectronAbundance`                     | *(optional)*                                                  |
| `/PartType0/StarFormationRate`                     | *(optional)*                                                  |
| `/PartType0/GFM_Metals`                            | *(optional)*, shape `[N, n_metals]`                           |

If no `snapdir_NNN/` directory exists, PROFI also tries the single-file path `output/snap_NNN.hdf5`.

If the per-subfile FoF catalogue is absent, PROFI also tries `output/fof_subhalo_tab_NNN.hdf5`.

---

## Parallelisation

PROFI uses **OpenMP** to parallelise the inner halo loop.  Control the thread count with:

```bash
export OMP_NUM_THREADS=8
./PROFI params/params_MySim.txt 99 99 0 0
```

For snapshots split into many FoF subfiles, the recommended strategy is to launch **one PROFI process per subfile** in parallel (shell `&`/`wait` or a SLURM job array) so that all subfiles run concurrently.

A ready-to-adapt SLURM script is provided in [`jobs/j_MySim_snap99.sh`](jobs/j_MySim_snap99.sh). It demonstrates the recommended pattern for 40 subfiles (0-39): subfile 0 gets more OpenMP threads, while subfiles 1-39 run concurrently with fewer threads. The script records every worker PID and checks its exit status. It waits for all workers, reports each failed subfile and its exit code, and exits with status 1 if any worker fails; the success message is printed only when all workers succeed.

Update the scheduler settings and marked `# <<<` values for your cluster. Adjust the subfile loop range for your snapshot and ensure the log directory exists.

The Python postprocessing scripts (Steps 2–4) use `ProcessPoolExecutor` to process multiple snapshots in parallel. Set `n_workers` in the configuration section at the top of each script.

---

## Contact

M. Reza Ayromlou — ayromlou@gmail.com

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for the full license text.
