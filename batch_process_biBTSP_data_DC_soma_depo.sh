#!/bin/bash
cd $HOME/PycharmProjects/BTSP

for cell_id in 32 33 34 35 36 43
do
    export cell_id
    python process_biBTSP_data_DC_soma.py --cell-id=$cell_id --plot=0 --export \
        --config-file-path=config/process_biBTSP_data_DC_soma_depo_config.yaml \
        --data-dir=data/20190825_biBTSP_DC_soma_depo --export-file-path=data/20200430_biBTSP_data_DC_soma_depo.h5
done
