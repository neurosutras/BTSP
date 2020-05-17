#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=optimize_biBTSP_synthetic_hybrid_alt2_step1"$DATE"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/src/BTSP/logs/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/src/BTSP/logs/$JOB_NAME.%j.e
#SBATCH -p development
#SBATCH -N 8
#SBATCH -n 448
#SBATCH -t 2:00:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $SCRATCH/src/BTSP

ibrun -n 448 python3 -m nested.optimize --config-file-path=config/optimize_biBTSP_synthetic_hybrid_alt2_step1_config.yaml \
    --output-dir=data --pop_size=200 --max_iter=50 --path_length=3 --disp
EOT
