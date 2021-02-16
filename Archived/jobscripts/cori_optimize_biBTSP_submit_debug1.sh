#!/bin/bash -l

cd $HOME/BTSP

for i in 11 28 29 30 31;  # 8;
do
    bash jobscripts/cori_optimize_biBTSP_cli_debug.sh $i 7 $1 $2
done
#
#for i in 15 18 4 5 7 9;  # 1
#do
#    bash jobscripts/cori_optimize_biBTSP_cli_debug.sh $i 13 $1 $2
#done
