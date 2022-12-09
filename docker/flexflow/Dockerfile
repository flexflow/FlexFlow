ARG FF_GPU_BACKEND "cuda"
FROM flexflow-environment-$FF_GPU_BACKEND:latest

LABEL org.opencontainers.image.source=https://github.com/flexflow/FlexFlow
LABEL org.opencontainers.image.description="FlexFlow container"

# Copy FlexFlow repository
RUN mkdir FlexFlow
ENV FF_HOME /usr/FlexFlow
WORKDIR ${FF_HOME}
COPY . .

# Args to build FlexFlow
ARG BUILD_CONFIGS
ARG N_BUILD_CORES

# Build and install C++ and Python versions of FlexFlow
RUN mkdir -p build && cd build && \
    eval "$BUILD_CONFIGS" ../config/config.linux && \
    make -j $N_BUILD_CORES && \
    eval "$BUILD_CONFIGS" ../config/config.linux && \
    make install && \
    ldconfig

ENTRYPOINT ["/bin/bash"]
