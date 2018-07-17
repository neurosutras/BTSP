#!/bin/bash -l

#SBATCH -J optimize_BTSP_CA1_cell4_20180501
#SBATCH -o /global/cscratch1/sd/aaronmil/BTSP/logs/optimize_BTSP_CA1_cell4_20180501.%j.o
#SBATCH -e /global/cscratch1/sd/aaronmil/BTSP/logs/optimize_BTSP_CA1_cell4_20180501.%j.e
#SBATCH -q regular
#SBATCH -N 13
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 12:00:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x
cd $HOME/BTSP
export cell_id=4

srun -N 13 -n 416 -c 2 --cpu_bind=cores python -m nested.optimize --config-file-path=config/optimize_BTSP_CA1_cell"$cell_id"_config.yaml --pop-size=200 --max-iter=50 --path-length=3 --disp --output-dir=$SCRATCH/BTSP --framework=pc --export --label=v12_cell$cell_id --cell_id=$cell_id
