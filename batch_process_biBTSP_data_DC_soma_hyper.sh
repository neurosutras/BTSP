#!/bin/bash
cd $HOME/PycharmProjects/BTSP

for ((cell_id=37;cell_id<=41;cell_id++))
do
    export cell_id
    for induction in 2
    do
        export induction
        python process_biBTSP_data_DC_soma.py --cell-id=$cell_id --induction=$induction --plot --export \
            --config-file-path=config/process_biBTSP_data_DC_soma_hyper_config.yaml \
            --data-dir=data/20200306_biBTSP_DC_soma_hyper --export-file-path=data/20200316_biBTSP_data_DC_soma_hyper.h5
    done
done
