# Flexflow base image
FROM flexflow:latest

ENV LANG C.UTF-8
ARG DEBIAN_FRONTEND=noninteractive

# Install basic packages
RUN rm -rf /var/lib/apt/lists/* \
	/etc/apt/sources.list.d/cuda.list \
	/etc/apt/sources.list.d/nvidia-ml.list && \
	apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    	apt-get install -y --no-install-recommends \
        software-properties-common && \
    apt-get update && \
    DEBIAN_FRONTEND=noninteractive \
    	apt-get install -y --no-install-recommends \
        build-essential \
        apt-utils \
        ca-certificates \
        libssl-dev \
        curl \
        unzip \
        htop \
	    nano && \
    DEBIAN_FRONTEND=noninteractive \
        apt-get install -y software-properties-common && \
        add-apt-repository ppa:ubuntu-toolchain-r/test && \
        apt-get update -y && \
        apt-get upgrade -y libstdc++6

# Install CPU-only Pytorch
RUN /opt/conda/bin/conda install pytorch==1.9.0 torchvision==0.10.0 torchaudio==0.9.0 cpuonly -c pytorch
# Install Hugging-face/MT5 related packages
RUN /opt/conda/bin/conda install -c conda-forge pandas numpy transformers=4.16.2 sentencepiece
# Install packages required by other example applications
RUN pip install onnx tensorflow keras2onnx
