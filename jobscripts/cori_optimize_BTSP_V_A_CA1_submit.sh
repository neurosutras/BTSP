#!/bin/bash -l

cd $HOME/BTSP

for i in 11 28 29 30 31 8;
do
    bash jobscripts/cori_optimize_BTSP_V_A_CA1_cli.sh $i 7
done

for i in 1 15 18 4 5 7 9;
do
    bash jobscripts/cori_optimize_BTSP_V_A_CA1_cli.sh $i 13
done
