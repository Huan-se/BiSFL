#!/bin/bash

echo "Starting script A..."
nohup bash run_origin.sh > run_A.log 2>&1 &

echo "Starting script B..."
nohup ./run_pro.sh > run_B.log 2>&1 &

echo "All scripts started."