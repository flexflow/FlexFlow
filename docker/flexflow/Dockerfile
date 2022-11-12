FROM flexflow-environment:latest

# Copy FlexFlow repository
RUN mkdir FlexFlow
ENV FF_HOME /usr/FlexFlow
WORKDIR ${FF_HOME}
COPY . .

# Args to build FlexFlow
ARG FF_GPU_BACKEND "cuda"
ARG N_BUILD_CORES
ARG FF_CUDA_ARCH

# Build and install C++ and Python versions of FlexFlow
RUN mkdir -p build && cd build && \
    ../config/config.linux && \
    make -j $N_BUILD_CORES && \
    ../config/config.linux && \
    make install

ENTRYPOINT ["/bin/bash"]
