#!/bin/bash
cd $HOME/PycharmProjects/BTSP

for ((cell_id=1;cell_id<=31;cell_id++))
do
    export cell_id
    for induction in 1 2
    do
        export induction
        python process_biBTSP_data.py --cell-id=$cell_id --induction=$induction --plot=0 --export \
            --export-file-path=data/20190717_biBTSP_data_before_LSA.bak
    done
done
