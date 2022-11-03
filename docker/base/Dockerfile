FROM nvidia/cuda:11.1.1-devel-ubuntu18.04

RUN apt-get update && apt-get install -y --no-install-recommends wget sudo binutils git zlib1g-dev

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

# Optionally install HIP dependencies
# Note that amd's docs say to also install the `hip-runtime-nvidia` package. This
# package attempts to re-install cuda even though cuda is already installed
# in the container. It also attempts to install packages for a graphical install.
# For our container, we don't need `hip-runtime-nvidia`
ARG FF_GPU_BACKEND "cuda"
RUN  if [ "$FF_GPU_BACKEND" = "hip_cuda" ] || [ "$FF_GPU_BACKEND" = "hip_rocm" ]; then \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Installing HIP dependencies"; \
        wget https://repo.radeon.com/amdgpu-install/latest/ubuntu/bionic/amdgpu-install_22.20.50200-1_all.deb; \
        apt-get install -y ./amdgpu-install_22.20.50200-1_all.deb; \
        rm ./amdgpu-install_22.20.50200-1_all.deb; \
        amdgpu-install -y --usecase=hip,rocm --no-dkms; \
        apt-get install -y hip-dev hipblas miopen-hip rocm-hip-sdk; \
     else \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Skipping installing HIP dependencies"; \
     fi

RUN rm -rf /var/lib/apt/lists/*

RUN /opt/conda/bin/conda install cmake make pillow
RUN /opt/conda/bin/conda install -c conda-forge numpy keras-preprocessing pybind11 cmake-build-extension

ENV PATH /opt/conda/bin:$PATH
ENV CUDNN_DIR /usr/local/cuda
ENV CUDA_DIR /usr/local/cuda

ENV FF_HOME /usr/FlexFlow
WORKDIR ${FF_HOME}

COPY . .

ARG N_BUILD_CORES
ARG FF_CUDA_ARCH

# Build and install C++ and Python versions of FlexFlow
RUN mkdir -p build && cd build && \
    ../config/config.linux && \
    make -j $N_BUILD_CORES && \
    ../config/config.linux && \
    make install

ENTRYPOINT ["/bin/bash"]
