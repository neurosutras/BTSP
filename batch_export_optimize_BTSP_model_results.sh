#!/bin/bash
cd $HOME/PycharmProjects/BTSP

export export_file_path=$1
echo $export_file_path
for cell_id in 4 5 7 9 11 15 18 28 29 30 31 # 1 8
do
    export cell_id
    python optimize_BTSP_CA1.py --debug --export --export-file-path=$export_file_path --cell_id=$cell_id --verbose=2
done
