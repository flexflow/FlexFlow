FROM nvidia/cuda:11.1.1-devel-ubuntu18.04


RUN apt-get update && apt-get install -y --no-install-recommends wget sudo binutils git zlib1g-dev lsb-release nano libhdf5-dev && \
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

# Optionally install HIP dependencies
# Note that amd's docs say to also install the `hip-runtime-nvidia` package. This
# package attempts to re-install cuda even though cuda is already installed
# in the container. It also attempts to install packages for a graphical install.
# For our container, we don't need `hip-runtime-nvidia`
ARG FF_GPU_BACKEND "cuda"
RUN  if [ "$FF_GPU_BACKEND" = "hip_cuda" ] || [ "$FF_GPU_BACKEND" = "hip_rocm" ]; then \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Installing HIP dependencies"; \
        # Get latest version of amdgpu-install script. We have to do this because new versions are rolled out
        # relatively often, and previous versions are removed, so only the latest version is available.
        # Similar approach as: https://stackoverflow.com/questions/22510705/get-the-latest-download-link-programmatically
        latest_version=$(curl 'http://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/' | \
        grep -oP 'href="amdgpu-install_\K[0-9]+\.[0-9]+\.[0-9]+\-[0-9]+' | \
        sort -t. -rn -k1,1 -k2,2 -k3,3 | head -1); \
        script_name="amdgpu-install_${latest_version}_all.deb"; \
        script_url="https://repo.radeon.com/amdgpu-install/latest/ubuntu/focal/${script_name}"; \
        eval wget "$script_url"; \
        apt-get install -y "./${script_name}"; \
        rm "./${script_name}"; \
        amdgpu-install -y --usecase=hip,rocm --no-dkms; \
        apt-get install -y hip-dev hipblas miopen-hip rocm-hip-sdk; \
    else \
        echo "FF_GPU_BACKEND: ${FF_GPU_BACKEND}. Skipping installing HIP dependencies"; \
    fi
RUN rm -rf /var/lib/apt/lists/*

# Set env varas
ENV PATH /opt/conda/bin:$PATH
ENV CUDNN_DIR /usr/local/cuda
ENV CUDA_DIR /usr/local/cuda
ENV FF_HOME /usr/FlexFlow

# Install python packages and other dependencies
RUN conda install -c conda-forge cmake make pillow cmake-build-extension pybind11 numpy pandas keras-preprocessing 

# Copy FlexFlow repository
WORKDIR ${FF_HOME}
COPY . .

# Args to build FlexFlow
ARG N_BUILD_CORES
ARG FF_CUDA_ARCH

# Build and install C++ and Python versions of FlexFlow
RUN mkdir -p build && cd build && \
    ../config/config.linux && \
    make -j $N_BUILD_CORES && \
    ../config/config.linux && \
    make install

ENTRYPOINT ["/bin/bash"]
