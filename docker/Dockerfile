FROM nvidia/cuda:11.1-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends wget sudo binutils git zlib1g-dev && \
    rm -rf /var/lib/apt/lists/*

RUN wget -c http://developer.download.nvidia.com/compute/redist/cudnn/v8.0.5/cudnn-11.1-linux-x64-v8.0.5.39.tgz && \
    tar -xzf cudnn-11.1-linux-x64-v8.0.5.39.tgz -C /usr/local && \
    rm cudnn-11.1-linux-x64-v8.0.5.39.tgz && \
    ldconfig

RUN wget -c https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    mv Miniconda3-latest-Linux-x86_64.sh ~/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x ~/Miniconda3-latest-Linux-x86_64.sh && \
    ~/Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda && \
    rm ~/Miniconda3-latest-Linux-x86_64.sh && \
    /opt/conda/bin/conda upgrade --all && \
    /opt/conda/bin/conda install conda-build conda-verify && \
    /opt/conda/bin/conda clean -ya

RUN /opt/conda/bin/conda install cmake make pillow
RUN /opt/conda/bin/conda install -c conda-forge protobuf=3.9 numpy keras-preprocessing

ENV PATH /opt/conda/bin:$PATH
ENV CUDNN_DIR /usr/local/cuda
ENV CUDA_DIR /usr/local/cuda
ENV PROTOBUF_DIR /opt/conda/pkgs/libprotobuf-3.9.2-h8b12597_0
ENV LD_LIBRARY_PATH $PROTOBUF_DIR/lib:$LD_LIBRARY_PATH

RUN cd /usr && \
    git clone --recursive https://github.com/flexflow/FlexFlow.git

ENV FF_HOME /usr/FlexFlow
ENV LG_RT_DIR /usr/FlexFlow/legion/runtime

RUN cd /usr/FlexFlow/python && \
    make -j4

WORKDIR /usr/FlexFlow
