FROM hotline_base_image:latest

# Install model specific dependencies.
WORKDIR /home/ubuntu/algorithmic-efficiency
RUN git checkout GNN-annotated-hotline-mlsys
RUN pip install -e '.[ogbg]'

CMD rm -rf experiment_dir; python3 submission_runner.py \
    --framework=pytorch \
    --workload=ogbg \
    --experiment_dir=experiment_dir \
    --experiment_name=baseline \
    --submission_path=reference_algorithms/development_algorithms/ogbg/ogbg_pytorch/submission.py \
    --tuning_search_space=reference_algorithms/development_algorithms/ogbg/tuning_search_space.json \
    --data_dir=/home/ubuntu/dataset/ogbg_molpcba \
    2>&1 | tee /home/ubuntu/hotline/results/gnn_hotline_analyze.log

