#!/bin/bash
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-gpu=32        
#SBATCH --time=0-6:00:00
#SBATCH --mem=200GB
#SBATCH -e /home/nkoreub/output_dir/log_%j.err
#SBATCH -o /home/nkoreub/output_dir/log_%j.out

echo "================ Job Info ================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"

echo "================ Environment ============="
echo "CPUs allocated: $NSLOTS"
echo "Python version: $(python --version)"

echo "================ Starting Job ============"
echo "Config file: $1"

export HF_HOME=/fast/AG_Kainmueller/nkoreub/hf_cache
N_CPUS=10
OMP_NUM_THREADS=$N_CPUS PYTHONPATH=/fast/AG_Kainmueller/nkoreub/Pathology-Foundation-Model-Benchmark python src/benchmark/run/train.py --config $1 --job_id ${SLURM_JOB_ID}
