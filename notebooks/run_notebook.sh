$ qrsh
Running inside SGE, jobid 4077341, 1 slot.
$ echo ${SGE_INTERACT_PORT}
18044
$ -l gpu=1
$ -l m_mem_free=40G
$ -l cuda_name="A40-PCIE-45G"
$ -l h_rt=24:00:00
$ -cwd
$ -V
$ -l h_rt=7:00:00
$ -A kainmueller
### $ -pe mpi 2
### $ hostname 
### max106.mdc-berlin.net
$ jupyter notebook --ip ${HOSTNAME} --port ${SGE_INTERACT_PORT} --no-browser 