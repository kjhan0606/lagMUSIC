#!/bin/bash
#SBATCH -J music_2lpt_L10_pair
#SBATCH -p normal
#SBATCH -N 1
#SBATCH -n 4
#SBATCH -c 16
#SBATCH -t 04:00:00
#SBATCH -o /home/kjhan/BACKUP/LagMUSIC/music/slurm-2lpt-L10-pair-%j.out
#SBATCH -e /home/kjhan/BACKUP/LagMUSIC/music/slurm-2lpt-L10-pair-%j.err

set -uo pipefail

module list 2>&1 || true
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}
export OMP_PLACES=cores
export OMP_PROC_BIND=close

WORK=/home/kjhan/BACKUP/LagMUSIC/music
cd "$WORK"

echo "== host: $(hostname)  date: $(date)"
echo "== nranks=${SLURM_NTASKS}  threads_per_rank=${OMP_NUM_THREADS}"
echo "== 2LPT L=10 (1024^3) baseline vs slab"
echo "== rank-0 baseline mem ~7 padded buffers x 9.4 GB = ~66 GB"
echo "== rank-0 slab    mem ~2 padded buffers x 9.4 GB = ~19 GB"
echo

echo "==== BASELINE (slab_2lpt_unigrid=no) ===="
T0=$SECONDS
mpirun -np "${SLURM_NTASKS}" "$WORK/build/MUSIC" \
    "$WORK/test_unigrid_2lpt_L10_baseline.conf" \
    > "$WORK/test_unigrid_2lpt_L10_baseline.out" 2>&1
RC_B=$?
echo "baseline rc=$RC_B  elapsed=$((SECONDS-T0))s"
grep -E 'lpt2-profile|sigma_v|FATAL|ERROR|out of memory|killed' \
    "$WORK/test_unigrid_2lpt_L10_baseline.out" || true
echo

echo "==== SLAB (slab_2lpt_unigrid=yes) ===="
T0=$SECONDS
mpirun -np "${SLURM_NTASKS}" "$WORK/build/MUSIC" \
    "$WORK/test_unigrid_2lpt_L10_slab.conf" \
    > "$WORK/test_unigrid_2lpt_L10_slab.out" 2>&1
RC_S=$?
echo "slab     rc=$RC_S  elapsed=$((SECONDS-T0))s"
grep -E 'lpt2-profile|sigma_v|FATAL|ERROR|out of memory|killed' \
    "$WORK/test_unigrid_2lpt_L10_slab.out" || true
echo

echo "== done: $(date)"
ls -lh /tmp/test_unigrid_2lpt_L10_*.hdf5 2>/dev/null || true
