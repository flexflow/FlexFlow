ARG ff_environment_base_image
FROM ${ff_environment_base_image}

LABEL org.opencontainers.image.source=https://github.com/flexflow/FlexFlow
LABEL org.opencontainers.image.description="FlexFlow environment container"

# Install basic dependencies
RUN apt-get update && apt-get install -y --no-install-recommends wget sudo binutils git zlib1g-dev lsb-release nano gdb libhdf5-dev jq && \
    rm -rf /var/lib/apt/lists/* /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list && \
	apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends software-properties-common && \
    apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends build-essential apt-utils \
    ca-certificates libssl-dev curl unzip htop && DEBIAN_FRONTEND=noninteractive \
        apt-get install -y software-properties-common && \
        add-apt-repository ppa:ubuntu-toolchain-r/test && \
        apt-get update -y && \
        apt-get upgrade -y libstdc++6

# Install Python3 with Miniconda
ARG python_version "latest"
#RUN MINICONDA_SCRIPT_NAME=Miniconda3-latest-Linux-x86_64.sh; \
RUN MINICONDA_SCRIPT_NAME=Miniconda3-py311_23.5.2-0-Linux-x86_64.sh; \
    if [ "$python_version" != "3.8" ] && [ "$python_version" != "3.9" ] && [ "$python_version" != "3.10" ] && [ "$python_version" != "3.11" ] && [ "$python_version" != "latest" ]; then \
        echo "python_version '${python_version}' is not supported, please choose among {3.8, 3.9, 3.10, 3.11 or latest (default)}"; \
        exit 1; \
    fi; \
    if [ "${python_version}" = "3.8" ]; then \
        MINICONDA_SCRIPT_NAME=Miniconda3-py38_23.5.2-0-Linux-x86_64.sh; \
    elif [ "${python_version}" = "3.9" ]; then \
        MINICONDA_SCRIPT_NAME=Miniconda3-py39_23.5.2-0-Linux-x86_64.sh; \
    elif [ "${python_version}" = "3.10" ]; then \
        MINICONDA_SCRIPT_NAME=Miniconda3-py310_23.5.2-0-Linux-x86_64.sh; \
    elif [ "${python_version}" = "3.11" ]; then \
        MINICONDA_SCRIPT_NAME=Miniconda3-py311_23.5.2-0-Linux-x86_64.sh; \
    fi; \
    wget -c -q https://repo.continuum.io/miniconda/${MINICONDA_SCRIPT_NAME} && \
        mv ./${MINICONDA_SCRIPT_NAME} ~/${MINICONDA_SCRIPT_NAME} && \
        chmod +x ~/${MINICONDA_SCRIPT_NAME} && \
        bash ~/${MINICONDA_SCRIPT_NAME} -b -p /opt/conda && \
        rm ~/${MINICONDA_SCRIPT_NAME} && \
        /opt/conda/bin/conda upgrade --all && \
        /opt/conda/bin/conda install conda-build conda-verify && \
        /opt/conda/bin/conda clean -ya

# Optionally install HIP dependencies
# Note that amd's docs say to also install the `hip-runtime-nvidia` package. This
# package attempts to re-install cuda even though cuda is already installed
# in the container. It also attempts to install packages for a graphical install.
# For our container, we don't need `hip-runtime-nvidia`
ARG FF_GPU_BACKEND "cuda"
ARG hip_version "5.6"
ARG N_BUILD_CORES
# set MAKEFLAGS to speedup any dependency that uses make
ENV MAKEFLAGS "${MAKEFLAGS} -j${N_BUILD_CORES}"

RUN  if [ "$FF_GPU_BACKEND" = "hip_cuda" ] || [ "$FF_GPU_BACKEND" = "hip_rocm" ]; then \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Installing HIP dependencies"; \
        # Check that hip_version is one of 5.3,5.4,5.5,5.6
        if [ "$hip_version" != "5.3" ] && [ "$hip_version" != "5.4" ] && [ "$hip_version" != "5.5" ] && [ "$hip_version" != "5.6" ]; then \
            echo "hip_version '${hip_version}' is not supported, please choose among {5.3, 5.4, 5.5, 5.6}"; \
            exit 1; \
        fi; \
        # Compute script name and url given the version
        AMD_GPU_SCRIPT_NAME=amdgpu-install_5.6.50600-1_all.deb; \
        if [ "$hip_version" = "5.3" ]; then \
            AMD_GPU_SCRIPT_NAME=amdgpu-install_5.3.50300-1_all.deb; \
        elif [ "$hip_version" = "5.4" ]; then \
            AMD_GPU_SCRIPT_NAME=amdgpu-install_5.4.50400-1_all.deb; \
        elif [ "$hip_version" = "5.5" ]; then \
            AMD_GPU_SCRIPT_NAME=amdgpu-install_5.5.50500-1_all.deb; \
        fi; \
        AMD_GPU_SCRIPT_URL="https://repo.radeon.com/amdgpu-install/${hip_version}/ubuntu/focal/${AMD_GPU_SCRIPT_NAME}"; \
        # Download and install AMD GPU software with ROCM and HIP support
        wget $AMD_GPU_SCRIPT_URL; \
        apt-get install -y ./${AMD_GPU_SCRIPT_NAME}; \
        rm ./${AMD_GPU_SCRIPT_NAME}; \
        amdgpu-install -y --usecase=hip,rocm --no-dkms; \
        apt-get install -y hip-dev hipblas miopen-hip rocm-hip-sdk rocm-device-libs; \
        # Install protobuf dependencies
        apt-get update -y && sudo apt-get install -y pkg-config zip g++ zlib1g-dev autoconf automake libtool make; \
    else \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Skipping installing HIP dependencies"; \
    fi
RUN rm -rf /var/lib/apt/lists/*

# Set env vars
ENV PATH /opt/conda/bin:$PATH
ENV CUDNN_DIR /usr/local/cuda
ENV CUDA_DIR /usr/local/cuda

# Install python packages and other dependencies
RUN conda install -c conda-forge cmake make pillow cmake-build-extension pybind11 numpy pandas keras-preprocessing
# Install CPU-only Pytorch and related dependencies
RUN conda install pytorch torchvision torchaudio -c pytorch
RUN conda install -c conda-forge onnx transformers>=4.31.0 sentencepiece einops
RUN pip3 install tensorflow notebook

# Install Rust
RUN curl https://sh.rustup.rs -sSf | sh -s -- -y
ENV PATH /root/.cargo/bin:$PATH

ENTRYPOINT ["/bin/bash"]
