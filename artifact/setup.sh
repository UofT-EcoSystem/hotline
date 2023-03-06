

# So datasets can be reused across runs, we mount them into the container.
DATASET="${DATASET:=$(pwd)/dataset}"  # If variable not set or null, set it to dataset in working directory.
mkdir -p $DATASET && chmod ugo+rw $DATASET > /dev/null 2>&1
export MOUNTS="-v $DATASET:/home/ubuntu/dataset"

# Put the results in the right place for the UI container to pick them up.
RESULTS="${RESULTS:=$(pwd)/results}"  # If variable not set or null, set it to results in working directory.
mkdir -p $RESULTS && chmod ugo+rw $RESULTS > /dev/null 2>&1
export MOUNTS="$MOUNTS -v $RESULTS:/home/ubuntu/hotline/results"
export MOUNTS="$MOUNTS -v $RESULTS/ui/src/results:/home/ubuntu/hotline/ui/src/results"
export MOUNTS="$MOUNTS -v $RESULTS/ui/dist/traces/results:/home/ubuntu/hotline/ui/dist/traces/results"
mkdir -p $RESULTS/ui/dist/traces/results && chmod ugo+rw $RESULTS/ui/dist/traces/results > /dev/null 2>&1
mkdir -p $RESULTS/ui/src/results && chmod ugo+rw $RESULTS/ui/src/results > /dev/null 2>&1

export EXTRA_ARGS="--env HOTLINE_DATASET_DIR=/home/ubuntu/dataset --env CUDA_VISIBLE_DEVICES=0"

echo "MOUNTS: $MOUNTS"
echo "EXTRA_ARGS: $EXTRA_ARGS"
echo "EXTRA_MOUNTS: $EXTRA_MOUNTS"
echo "DATASET DIR: $DATASET"
echo "RESULTS DIR: $RESULTS"

function run {
    pushd $1 > /dev/null
    docker run --network=host --gpus all --ipc=host --shm-size 64G $(echo $MOUNTS) $(echo $EXTRA_MOUNTS) `echo $EXTRA_ARGS` --rm -it hotline_$1_image $2
    popd > /dev/null
}

function build {
  # $1 is the name of the directory
  # $2 optional, if present, is the additional build args
    pushd $1 > /dev/null
    docker build --network=host --build-arg UID=$(id -u) --build-arg MY_TOKEN=$MY_TOKEN -t hotline_$1_image $2 .
    popd > /dev/null
}
