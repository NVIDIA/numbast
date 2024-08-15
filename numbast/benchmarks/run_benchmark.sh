#!/bin/bash

NUMBAST_BENCH_KERN_REPETITION=1000

BENCH_NAME=test_arithmetic_bf16

PY_NAME=${BENCH_NAME}.py
GOLD_NAME=${BENCH_NAME}_gold
GOLD_SRC_NAME=${GOLD_NAME}.cu

# Cleanup
rm -rf *.json *.nsys-rep *.sqlite $GOLD_NAME

# Compile gold
nvcc --gpu-architecture=sm_70 $GOLD_SRC_NAME -o $GOLD_NAME

# Prof gold
nsys profile --trace cuda --force-overwrite true -o gold.nsys-rep $GOLD_NAME

# Prof py
nsys profile --trace cuda --force-overwrite true -o py.nsys-rep python $PY_NAME

# # Analyze gold
nsys stats --report cuda_gpu_kern_sum --format json --output . gold.nsys-rep

# # Analyze py
nsys stats --report cuda_gpu_kern_sum --format json --output . py.nsys-rep
