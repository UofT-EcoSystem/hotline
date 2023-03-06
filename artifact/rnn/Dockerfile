FROM hotline_base_image:latest

# Install model specific dependencies.
RUN pip install install panda librosa Levenshtein
WORKDIR /home/ubuntu/
RUN git clone --recursive https://github.com/parlance/ctcdecode.git
WORKDIR /home/ubuntu/ctcdecode
RUN python3 setup.py install --user --quiet > /dev/null 2>&1

WORKDIR /home/ubuntu/algorithmic-efficiency
RUN git checkout RNN-hotline-mlsys

CMD bash -c 'set -o pipefail; cd ~/hotline; git pull; cd -; git pull; rm -rf experiment_dir; python3 submission_runner.py \
    --framework=pytorch \
    --workload=librispeech \
    --submission_path=reference_submissions/librispeech/librispeech_pytorch/submission.py \
    --tuning_search_space=reference_submissions/librispeech/tuning_search_space.json \
    --data_dir=/home/ubuntu/algorithmic-efficiency/dataset-librispeech100-small/work_dir/data \
    2>&1 | tee /home/ubuntu/hotline/results/rnn_hotline_analyze.log'
