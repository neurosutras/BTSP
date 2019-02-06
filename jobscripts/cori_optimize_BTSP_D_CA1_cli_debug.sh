#!/bin/bash -l
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J optimize_BTSP_D_CA1_cell"$1"_20190206
#SBATCH -o /global/cscratch1/sd/aaronmil/BTSP/logs/optimize_BTSP_D_CA1_cell"$1"_20190206.%j.o
#SBATCH -e /global/cscratch1/sd/aaronmil/BTSP/logs/optimize_BTSP_D_CA1_cell"$1"_20190206.%j.e
#SBATCH -q debug
#SBATCH -N 1
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x
cd $HOME/BTSP

srun -N 1 -n 32 -c 2 --cpu_bind=cores python -m nested.optimize \
    --config-file-path=config/optimize_BTSP_D_CA1_cli_config.yaml --analyze --disp --output-dir=$SCRATCH/BTSP \
    --label=$1 --cell_id=$1
EOT