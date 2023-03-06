FROM hotline_base_image:latest

# Install model specific dependencies.
WORKDIR /home/ubuntu/algorithmic-efficiency
RUN git checkout transformer-hotline-mlsys23
RUN pip install -e '.[wmt]'

CMD bash -c 'set -o pipefail; cd ~/hotline; git pull; cd -; git pull; rm -rf experiment_dir; python3 submission_runner.py \
    --framework=pytorch \
    --workload=wmt \
    --experiment_dir=experiment_dir \
    --experiment_name=baseline \
    --submission_path=reference_algorithms/development_algorithms/wmt/wmt_pytorch/submission.py \
    --tuning_search_space=reference_algorithms/development_algorithms/wmt/tuning_search_space.json \
    --data_dir=/home/ubuntu/dataset/wmt17_translate \
    2>&1 | tee /home/ubuntu/hotline/results/transformer_hotline_analyze.log'
