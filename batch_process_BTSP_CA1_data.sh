#!/bin/bash
cd $HOME/PycharmProjects/BTSP

for ((cell_id=1;cell_id<=31;cell_id++))
do
    export cell_id
    for induction in 1 2
    # for induction in 2
    do
        export induction
        python process_BTSP_CA1_data.py --cell-id=$cell_id --induction=$induction --plot=0 --export \
            --export-file-path=data/20180411_BTSP2_CA1_data_before_LSA.bak
    done
done
