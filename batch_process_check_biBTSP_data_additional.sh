#!/bin/bash
cd $HOME/src/BTSP

for cell_id in 37 44 45 46 47 48 50 51 52 53
do
    export cell_id
    python process_biBTSP_data.py --cell-id=$cell_id --plot=0 \
        --config-file-path=config/process_biBTSP_data_config.yaml \
        --data-dir=data/20200409_biBTSP_additional_cells
done
