#$ -cwd
#$ -V
#$ -e /home/jluesch/output_dir/log_$JOB_ID.err
#$ -o /home/jluesch/output_dir/log_$JOB_ID.out
#$ -l gpu=1
#$ -l cuda_memory=55G
#$ -pe smp 2
#$ -l m_mem_free=65G
#$ -l h_rt=20:00:00
#$ -A kainmueller


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

nvidia-smi
export HF_HOME=/fast/AG_Kainmueller/fabian/miniforge/hf_cache
N_CPUS=8
OMP_NUM_THREADS=$N_CPUS PYTHONPATH=/fast/AG_Kainmueller/jluesch/channel_dinov2 python src/benchmark/run/train.py --config $1 --job_id ${JOB_ID} --model_name $2
