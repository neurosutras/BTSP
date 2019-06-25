#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=optimize_biBTSP_D_45cm_cell"$1"_"$DATE"
export cores=$(($2 * 32))
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /global/cscratch1/sd/aaronmil/BTSP/logs/"$JOB_NAME".%j.o
#SBATCH -e /global/cscratch1/sd/aaronmil/BTSP/logs/"$JOB_NAME".%j.e
#SBATCH -q premium
#SBATCH -N $2
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 06:00:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $HOME/BTSP

srun -N $2 -n $cores -c 2 --cpu_bind=cores python -m nested.optimize \
    --config-file-path=config/optimize_biBTSP_D_45cm_cli_config.yaml --disp --output-dir=$SCRATCH/BTSP \
    --pop-size=200 --max-iter=50 --path-length=3 --disp --export --label=cell"$1" --cell_id=$1
EOT
