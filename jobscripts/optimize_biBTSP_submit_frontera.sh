#!/bin/bash -l

cd $WORK/BTSP

for i in 11 28 29 30 31 8 44 45 46 48 49;
do
    bash jobscripts/optimize_biBTSP_cli_frontera.sh $i 4 $1 $2
done

for i in 15 18 4 5 7 9 47 1;  # 1
do
    bash jobscripts/optimize_biBTSP_cli_frontera.sh $i 8 $1 $2
done
