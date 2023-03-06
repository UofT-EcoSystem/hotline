#!/bin/bash

source setup.sh
set -e
set -x

# Single GPU experiments
export EXTRA_ARGS="--env HOTLINE_DATASET_DIR=/home/ubuntu/dataset --env CUDA_VISIBLE_DEVICES=0"
# export EXTRA_ARGS=$EXTRA_ARGS" --env HOTLINE_BATCH_SIZE=4"
# run resnet
# run transformer

# 4 x GPU experiments
export EXTRA_ARGS="--env HOTLINE_DATASET_DIR=/home/ubuntu/dataset --env CUDA_VISIBLE_DEVICES=0,1,2,3"
run rnn
# export EXTRA_ARGS=$EXTRA_ARGS" --env HOTLINE_BATCH_SIZE=4"
# run resnet
# run transformer



