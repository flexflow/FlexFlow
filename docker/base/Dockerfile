FROM nvidia/cuda:11.1.1-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends wget sudo binutils git zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN wget -c -q http://developer.download.nvidia.com/compute/redist/cudnn/v8.0.5/cudnn-11.1-linux-x64-v8.0.5.39.tgz && \
    tar -xzf cudnn-11.1-linux-x64-v8.0.5.39.tgz -C /usr/local && \
    rm cudnn-11.1-linux-x64-v8.0.5.39.tgz && \
    ldconfig

RUN wget -c -q https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mv Miniconda3-latest-Linux-x86_64.sh ~/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/Miniconda3-latest-Linux-x86_64.sh && \
    ~/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm ~/Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda upgrade --all && \
    /opt/conda/bin/conda install conda-build conda-verify && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install cmake make pillow
RUN /opt/conda/bin/conda install -c conda-forge numpy keras-preprocessing pybind11 cmake-build-extension

ENV PATH /opt/conda/bin:$PATH
ENV CUDNN_DIR /usr/local/cuda
ENV CUDA_DIR /usr/local/cuda

# Clone the repo, invalidating the cache in case of new commits
ADD https://api.github.com/repos/flexflow/flexflow/git/refs/heads/master ff_version.json
RUN cd /usr && \
    git clone --recursive https://github.com/flexflow/FlexFlow.git
ENV FF_HOME /usr/FlexFlow
# overwrite configs file with those (potentially) modified by the user
COPY ./config /usr/FlexFlow/config

ARG n_build_cores

# Build and install C++ and Python versions of FlexFlow
RUN cd /usr/FlexFlow && mkdir -p build && cd build && \
    ../config/config.linux && \
    make -j $n_build_cores && \
    ../config/config.linux && \
    make install

ENTRYPOINT ["/bin/bash"]

WORKDIR /usr/FlexFlow
