FROM hotline_base_image:latest

# Install model specific dependencies.
WORKDIR /home/ubuntu/algorithmic-efficiency
RUN git checkout DLRM-annotated-hotline-mlsys
RUN pip install -e '.[criteo1tb]'
RUN git pull

# Inflate the included sample data from 100,000 to 600,000 rows, enough for a full sized batch. This dataset can repeat for multiple batches.
RUN cat algorithmic_efficiency/workloads/criteo1tb/day_0_0.csv >> algorithmic_efficiency/workloads/criteo1tb/day_0_1.csv
RUN cat algorithmic_efficiency/workloads/criteo1tb/day_0_0.csv >> algorithmic_efficiency/workloads/criteo1tb/day_0_1.csv
RUN cat algorithmic_efficiency/workloads/criteo1tb/day_0_0.csv >> algorithmic_efficiency/workloads/criteo1tb/day_0_1.csv
RUN cat algorithmic_efficiency/workloads/criteo1tb/day_0_0.csv >> algorithmic_efficiency/workloads/criteo1tb/day_0_1.csv
RUN cat algorithmic_efficiency/workloads/criteo1tb/day_0_0.csv >> algorithmic_efficiency/workloads/criteo1tb/day_0_1.csv
RUN cat algorithmic_efficiency/workloads/criteo1tb/day_0_0.csv >> algorithmic_efficiency/workloads/criteo1tb/day_0_1.csv
RUN cat algorithmic_efficiency/workloads/criteo1tb/day_0_0.csv >> algorithmic_efficiency/workloads/criteo1tb/day_0_1.csv
RUN cat algorithmic_efficiency/workloads/criteo1tb/day_0_0.csv >> algorithmic_efficiency/workloads/criteo1tb/day_0_1.csv
RUN cat algorithmic_efficiency/workloads/criteo1tb/day_0_0.csv >> algorithmic_efficiency/workloads/criteo1tb/day_0_1.csv
RUN cat algorithmic_efficiency/workloads/criteo1tb/day_0_0.csv >> algorithmic_efficiency/workloads/criteo1tb/day_0_1.csv

CMD rm -rf experiment_dir; python3 submission_runner.py \
    --framework=pytorch \
    --workload=criteo1tb \
    --experiment_dir=experiment_dir \
    --experiment_name=baseline \
    --submission_path=reference_algorithms/development_algorithms/criteo1tb/criteo1tb_pytorch/submission.py \
    --tuning_search_space=reference_algorithms/development_algorithms/criteo1tb/tuning_search_space.json \
    --data_dir=/home/ubuntu/algorithmic-efficiency/algorithmic_efficiency/workloads/criteo1tb \
    2>&1 | tee /home/ubuntu/hotline/results/dlrm_hotline_analyze.log
    # --data_dir=/home/ubuntu/dataset/Criteo1TB_DLRM

# head -100000 day_0 > out.csv
# ls -sh out.csv  # 32 MB
# cp out.csv /home/dans/algorithmic-efficiency/algorithmic_efficiency/workloads/librispeech/day_0_0.csv
