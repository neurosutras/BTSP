#!/bin/bash
cd $HOME/PycharmProjects/BTSP

for ((cell_id=32;cell_id<=36;cell_id++))
do
    export cell_id
    for induction in 1
    do
        export induction
        python process_biBTSP_data.py --cell-id=$cell_id --induction=$induction --plot=0 --export \
            --config-file-path=config/process_biBTSP_data_soma_DC_config.yaml \
            --data-dir=data/20190825_biBTSP_DC_soma --export-file-path=data/20190825_biBTSP_data_soma_DC_before_LSA.bak
    done
done
