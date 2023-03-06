# Usage:
# docker build --build-arg MY_TOKEN=$MY_TOKEN --network=host -t hotline_base_image .
#
# Dev: (don't cache anything)
# git commit -am 'wip'; git push; sleep 0.2 && docker build --no-cache --build-arg MY_TOKEN=$MY_TOKEN --network=host -t hotline_base_image . && docker run --network=host --gpus all --ipc=host --shm-size 8G --rm -it hotline_base_image
#
# Debug:
# docker run --network=host --gpus all --ipc=host --shm-size 8G --rm -it hotline_base_image /bin/bash

FROM nvidia/cuda:11.2.0-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive

# Fix apt-get update error for nvidia apt repo.
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub

RUN apt-get update --fix-missing

RUN apt-get install -y bash \
                  build-essential \
                  git \
                  curl \
                  wget \
                  libssl-dev \
                  ca-certificates \
                  gnupg \
                  apt-transport-https \
                  python3 \
                  python3-pip \
                  htop \
                  tree \
                  bash-completion \
                  vim \
                  zip \
                  unzip \
                  tmux \
                  less \
                  sudo

RUN python3 -m pip install --no-cache-dir --upgrade pip

# Add user.
RUN echo 'root:password' | chpasswd
ARG UID=1001
ENV UID=${UID}
RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u $UID ubuntu
RUN echo 'ubuntu:password' | chpasswd
USER ubuntu
ENV PATH="/home/ubuntu/.local/bin:${PATH}"

# Install shared ML models (GNN, DLRM, Transformer, etc.)
WORKDIR /home/ubuntu
RUN git clone https://github.com/UofT-EcoSystem/algorithmic-efficiency.git
RUN git config --global --add safe.directory /home/ubuntu/algorithmic-efficiency
WORKDIR /home/ubuntu/algorithmic-efficiency
RUN pip3 install -e '.[jax_cpu]'
RUN pip3 install -e '.[pytorch_gpu]' -f 'https://download.pytorch.org/whl/torch_stable.html'
RUN pip3 install torch==1.12.0+cu113 torchvision==0.13.0+cu113 -f 'https://download.pytorch.org/whl/torch_stable.html' 
RUN pip3 install -e '.[full]'
# Full is missing a few things.
RUN pip3 install \
            absl-py==1.0.0 \
            numpy==1.21.6 \
            pandas==1.3.5 \
            protobuf==3.20.* \
            six==1.16.0 \
            tensorflow==2.9.0 \
            tensorflow_datasets==4.4.0 \
            tensorflow_probability==0.17.0 \
            tensorflow_addons==0.18.0 \
            gputil==1.4.0 \
            psutil==5.9.1 \
            clu==0.0.7
# Need this version of jax for the current state of hotline code.
RUN pip3 install \
            jax==0.2.28 \
            jaxlib==0.1.76 \
            -f https://storage.googleapis.com/jax-releases/jax_releases.html

# Install rust package manager (needed for python package "orjson")
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="$PATH:/home/ubuntu/.cargo/bin"

# Grab the code.
WORKDIR /home/ubuntu
RUN git clone https://github.com/UofT-EcoSystem/hotline.git
RUN git config --global --add safe.directory /home/ubuntu/hotline
WORKDIR /home/ubuntu/hotline
# Install python packages.
RUN pip3 install --no-cache-dir -e .

ENV HOTLINE_OUTPUT_DIR='/home/ubuntu/hotline'

CMD python3 examples/import.py