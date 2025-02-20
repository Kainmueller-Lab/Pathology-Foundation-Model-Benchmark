#!/bin/bash
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-gpu=10
#SBATCH --nodes=1
#SBATCH -e /home/jluesch/output_dir/log_%j.err
#SBATCH -o /home/jluesch/output_dir/log_%j.out
#SBATCH --time 0-10:00:00
#SBATCH --nodelist=maxg[10,20]
#SBATCH --account=kainmueller
#SBATCH --mem=120G
#SBATCH --partition=gpu
#SBACTH --gres=gpu_memory:45G
#SBATCH --gres=gpu:1

echo "================ Job Info ================"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"

echo "================ Environment ============="
echo "CPUs allocated: $NSLOTS"
echo "Python version: $(python --version)"

echo "================ Starting Job ============"
echo "Config file: $1"
echo "Model name: $2"

nvidia-smi
export HF_HOME=/fast/AG_Kainmueller/fabian/miniforge/hf_cache
N_CPUS=10
OMP_NUM_THREADS=$N_CPUS PYTHONPATH=/fast/AG_Kainmueller/jluesch/Pathology-Foundation-Model-Benchmark python src/benchmark/train.py --config $1 --job_id ${SLURM_JOB_ID} --model_name $2
