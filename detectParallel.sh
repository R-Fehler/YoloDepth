#!/bin/bash

# This script runs parallel inference on a img folder containing imgs in a filename sequence from START...END.png
# needs one full cuda instance of model per process, 3 is max on rtx 2080 etc
source ~/venv-yolo/bin/activate
# NUM_CORES=$(nproc --all)
# NUM_CORES=$(($NUM_CORES/2))
NUM_CORES=3
START=500
END=1000
UNIT=$((($END-$START)/$NUM_CORES))
echo "UNIT is $UNIT NUMCORES is $NUM_CORES"
for i in $(seq 1 $NUM_CORES); do
NEXT=$(($START+$UNIT)) 
# generate leading zeros for the filename when necesarry
STARTFILENAME=$(printf "%010d" $START)
ENDFILENAME=$(printf "%010d" $NEXT)
# WARNING, change the Input Folder containing the imgs apropriatly
eval python detect.py OstringDepthDataset/imgs/{$STARTFILENAME..$ENDFILENAME}.png &> logs/$START.log &
START=$NEXT
done