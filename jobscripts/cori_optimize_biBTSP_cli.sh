#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=optimize_biBTSP_"$3"_cell"$1"_"$DATE"
export cores=$(($2 * 32))
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /global/cscratch1/sd/aaronmil/BTSP/logs/"$JOB_NAME".%j.o
#SBATCH -e /global/cscratch1/sd/aaronmil/BTSP/logs/"$JOB_NAME".%j.e
#SBATCH -q regular
#SBATCH -N $2
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 06:00:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $HOME/BTSP

srun -N $2 -n $cores -c 2 --cpu-bind=cores python -m mpi4py.futures -m nested.optimize \
    --config-file-path=config/optimize_biBTSP_"$3"_cli_config.yaml --disp --output-dir=$SCRATCH/BTSP \
    --pop_size=200 --max_iter=50 --path_length=3 --disp --export --label="$4"cm_cell"$1" --cell_id=$1 --framework=mpi \
    --input_field_width=$4
EOT
