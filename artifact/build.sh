#!/bin/bash -e

source setup.sh
set -e
set -x

build base
build resnet
build rnn
build transformer
# build gnn
# build dlrm
# build stable-diffusion
build ui
