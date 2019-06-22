#!/bin/bash
cd $HOME/PycharmProjects/BTSP

for ((cell_id=1;cell_id<=31;cell_id++))
do
    export cell_id
    for induction in 1 2
    do
        export induction
        python get_LSA_weights_BTSP_CA1.py --cell-id=$cell_id --induction=$induction --plot=1 --export \
            --data-file-path=data/20180329_BTSP2_CA1_data.hdf5
    done
done
