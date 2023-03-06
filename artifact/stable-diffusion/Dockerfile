FROM hotline_base_image:latest

# Grab the code.
WORKDIR /home/ubuntu
RUN git clone https://danielsnider:$MY_TOKEN@github.com/danielsnider/stable-diffusion-pytorch.git
WORKDIR /home/ubuntu/stable-diffusion-pytorch

# Install python packages.
RUN pip3 install --no-cache-dir -r requirements.txt

# Grab the pre-trained model (4 GB)
WORKDIR /home/ubuntu/stable-diffusion-pytorch
RUN wget --progress=bar:force:noscroll https://huggingface.co/jinseokim/stable-diffusion-pytorch-data/resolve/main/data.v20221029.tar
RUN tar -xf data.v20221029.tar
RUN rm data.v20221029.tar

ARG MY_TOKEN
ENV MY_TOKEN=${MY_TOKEN}

CMD git pull; python3 example_profiled.py 2>&1 | tee /home/ubuntu/hotline/results/stable-diffusion_hotline_analyze.log
