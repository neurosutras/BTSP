#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=optimize_biBTSP_synthetic_WD_"$DATE"
export cores=$((4 * 56))
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/logs/BTSP/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/BTSP/$JOB_NAME.%j.e
#SBATCH -p development
#SBATCH -N 4
#SBATCH -n $cores
#SBATCH -t 02:00:00
#SBATCH --mail-user=neurosutras@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $WORK/BTSP

ibrun -n $cores python3 -m nested.optimize \
    --config-file-path=config/optimize_biBTSP_synthetic_WD_config.yaml --disp \
    --output-dir=$SCRATCH/data/BTSP --hot-start \
    --storage-file-path=$SCRATCH/data/BTSP/20210217_155134_biBTSP_synthetic_WD_PopulationAnnealing_optimization_history.hdf5 \
    --pop_size=200 --max_iter=50 --path_length=3 --disp
EOT
