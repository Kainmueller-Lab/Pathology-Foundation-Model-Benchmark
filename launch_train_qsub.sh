#!/bin/bash
# -*- coding: utf-8 -*-
#$ -S /bin/bash
#$ -N patho_bench
#$ -e /home/nkoreub/output_dir
#$ -o /home/nkoreub/output_dir
#$ -l h_gpu=1
# Gets a GPU of min 30GB (=40G) (don't use in combination with -l cuda_name
#$ -l cuda_memory=60G
# request more than 1 CPU (-pe -> parallel environment; number (4) is number of cpus. Default is 1
#$ -pe smp 2
# request memory
#$ -l m_mem_free=65G
# runtime, max time is 96 hours - >96 hours, it puts it on the long queue -> takes longer to start running. (max might be higher now)
# Lorenz says the max is 144 hours now. I checked - it might be up until 336 hours now.
#$ -l h_rt=10:00:00
# get messages for error & abort
#$ -m ea
# Exclude maxg09 from this
# -l hostname=!maxg09


echo "================ Job Info ================"
echo "Job ID: $JOB_ID"
echo "Node: $HOSTNAME"
echo "Start time: $(date)"

echo "================ Environment ============="
echo "CPUs allocated: $NSLOTS"
echo "Python version: $(python --version)"

echo "================ Starting Job ============"
echo "Config file: $1"
echo "Model name: $2"

export HF_HOME=/fast/AG_Kainmueller/fabian/miniforge/hf_cache
N_CPUS=8
OMP_NUM_THREADS=$N_CPUS
export PYTHONPATH="${PYTHONPATH}:/fast/AG_Kainmueller/nkoreub/Pathology-Foundation-Model-Benchmark"

cd /fast/AG_Kainmueller/nkoreub/Pathology-Foundation-Model-Benchmark
python src/benchmark/train.py --config $1 --job_id ${JOB_ID} --model_name $2
