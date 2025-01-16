#$ -l gpu=1
#$ -l cuda_memory=60G
#$ -pe smp 4
#$ -cwd
#$ -V
#$ -e /home/jluesch/output_dir/log_$JOB_ID.err
#$ -o /home/jluesch/output_dir/log_$JOB_ID.out
#$ -l h_rt=10:00:00
#$ -A kainmueller

echo $JOB_ID
PYTHONPATH=/fast/AG_Kainmueller/jluesch/channel_dinov2 src/main.py


