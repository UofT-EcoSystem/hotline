FROM hotline_base_image:latest

WORKDIR /home/ubuntu/hotline

CMD bash -c 'set -o pipefail; git pull && python3 examples/example_resnet50.py 2>&1 | tee results/resnet_hotline_analyze.log'
