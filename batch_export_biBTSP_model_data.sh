#!/bin/bash
cd $HOME/src/BTSP

for cell_id in 11 28 29 30 31 8 44 45 46 48 49 15 18 4 5 7 9 47 1;
do
    export cell_id
    mpirun -n 7 python optimize_biBTSP_$1.py --cell_id=$cell_id --export \
        --config-file-path=$2 --param_file_path=$3 --model_key=$cell_id --export-file-path=$4 --verbose=1
done
