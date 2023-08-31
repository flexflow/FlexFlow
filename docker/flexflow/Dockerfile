ARG FF_GPU_BACKEND "cuda"
ARG gpu_backend_version ""
FROM flexflow-environment-$FF_GPU_BACKEND$gpu_backend_version:latest

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

# Create install directory if needed
RUN for pair in $BUILD_CONFIGS; do \
        key=${pair%%=*}; \
        value=${pair#*=}; \
        if [ "$key" = "INSTALL_DIR" ] && [ -n "$value" ]; then \
            mkdir -p "$value"; \
        fi; \
    done

# Build and install C++ and Python versions of FlexFlow
RUN mkdir -p build && cd build && \
    eval "$BUILD_CONFIGS" ../config/config.linux && \
    make -j $N_BUILD_CORES && \
    eval "$BUILD_CONFIGS" ../config/config.linux && \
    make install && \
    ldconfig

ENTRYPOINT ["/bin/bash"]
