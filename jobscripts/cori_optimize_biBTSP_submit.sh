#!/bin/bash -l

cd $HOME/BTSP

for i in 11 28 29 30 31 8;
do
    bash jobscripts/cori_optimize_biBTSP_cli.sh $i 7 $2 $3
done

for i in 15 18 4 5 7 9;  # 1
do
    bash jobscripts/cori_optimize_biBTSP_D_90cm_cli.sh $i 13 $2 $3
done
