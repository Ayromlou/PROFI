#!/bin/bash -l
#
# SLURM job script — sample for running PROFI on a single snapshot.
#
# This example processes snapshot 99 of a simulation that is split into
# 40 FOF subfiles (0-39).  Subfile 0 is typically the heaviest (it holds
# the most-massive halos), so it gets more OMP threads than the rest.
#
# Adapt the settings below (marked <<<) for your cluster and simulation.
#
#SBATCH -J PROFI_MySim_snap99
#SBATCH -o /path/to/job_logs/PROFI_MySim_snap99_%j.out   # <<<
#SBATCH -e /path/to/job_logs/PROFI_MySim_snap99_%j.err   # <<<
#SBATCH -D ./
#SBATCH --mail-user=your@email.com                        # <<<
#SBATCH --mail-type=ALL
#SBATCH --nodes=1
#SBATCH --cpus-per-task=96                                # <<<
#SBATCH --partition=your_partition                        # <<<
#SBATCH --time=23:59:00                                   # <<<
#SBATCH --mem=0

# ── Thread budget ──────────────────────────────────────────────────────────────
#
#   Total cores allocated: 96  (--cpus-per-task above)
#
#   Subfile 0 (heaviest, launched first):  N_HEAVY = 16 OMP threads
#   Subfiles 1-39 (39 jobs in parallel):   N_LIGHT =  2 OMP threads each
#   Peak concurrent usage: 16 + 39×2 = 94 cores  (within budget)
#
#   Rule of thumb:
#     N_HEAVY + (N_SUBFILES - 1) × N_LIGHT  <=  --cpus-per-task
#
# Adjust N_HEAVY, N_LIGHT, and the loop below to match your subfile count and
# core allocation.

PARAMS=../params/params_MySim.txt   # <<<  path to your PROFI parameter file
SNAP=99                              # <<<  snapshot number
N_HEAVY=16                           # <<<  OMP threads for subfile 0
N_LIGHT=2                            # <<<  OMP threads for all other subfiles

echo "======================================================"
echo " Job started at:  $(date)"
echo " Running on host: $(hostname)"
echo " SLURM job ID:    $SLURM_JOB_ID"
echo "======================================================"

# --- Subfile 0: heavier, launched first in background with more threads --------
pids=()
echo "[$(date)] Launching subfile 0 (OMP=${N_HEAVY})"
OMP_NUM_THREADS="${N_HEAVY}" ./../PROFI "$PARAMS" "$SNAP" "$SNAP" 0 0 0 &
pids[0]=$!

# --- Subfiles 1-39: all launched in parallel with fewer threads ---------------
# Every subfile is launched unconditionally — existing output is overwritten.
# Increase or decrease the range to match your simulation's subfile count.
echo "--- launching subfiles 1-39 at $(date) ---"
export OMP_NUM_THREADS="${N_LIGHT}"
for subfile in {1..39}; do
    ./../PROFI "$PARAMS" "$SNAP" "$SNAP" "$subfile" "$subfile" 0 &
    pids[$subfile]=$!
done

# Check each recorded PID, including workers that have already exited.
# Wait for every worker before reporting the final job status.
failed=0
for subfile in "${!pids[@]}"; do
    if wait "${pids[$subfile]}"; then
        echo "[$(date)] Subfile $subfile finished."
    else
        worker_status=$?
        printf 'ERROR: snapshot %s subfile %s exited with status %s.\n' \
            "$SNAP" "$subfile" "$worker_status" >&2
        failed=1
    fi
done

if (( failed )); then
    echo "--- snap $SNAP finished with failed workers at $(date) ---" >&2
else
    echo "--- snap $SNAP all done at $(date) ---"
fi
echo "======================================================"
echo " Job ended at: $(date)"
echo "======================================================"
exit "$failed"
