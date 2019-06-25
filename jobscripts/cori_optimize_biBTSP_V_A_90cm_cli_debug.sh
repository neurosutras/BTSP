#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=optimize_biBTSP_V_A_cell"$1"_"$DATE"_debug
export cores=$(($2 * 32))
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /global/cscratch1/sd/aaronmil/BTSP/logs/"$JOB_NAME".%j.o
#SBATCH -e /global/cscratch1/sd/aaronmil/BTSP/logs/"$JOB_NAME".%j.e
#SBATCH -q debug
#SBATCH -N $2
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $HOME/BTSP

source $HOME/.bash_profile_py3.ext

srun -N $2 -n $cores -c 2 --cpu_bind=cores python -m mpi4py.futures -m nested.optimize \
    --config-file-path=config/optimize_biBTSP_V_A_90cm_cli_config.yaml --disp --output-dir=$SCRATCH/BTSP \
    --pop-size=200 --max-iter=2 --path-length=1 --disp --export --label=cell"$1" --cell_id=$1 --framework=mpi
EOT
