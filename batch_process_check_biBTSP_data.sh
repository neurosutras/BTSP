#!/bin/bash
cd $HOME/src/BTSP

for ((cell_id=1;cell_id<=31;cell_id++))
do
    export cell_id
    python process_biBTSP_data.py --cell-id=$cell_id --plot=0 \
        --config-file-path=config/process_biBTSP_data_config.yaml \
        --data-dir=data/BTSP
done
