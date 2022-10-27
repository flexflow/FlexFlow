# Acknowledged to https://testdriven.io/blog/github-actions-docker/

FROM ubuntu:latest

ARG RUNNER_VERSION="2.298.2"

RUN DEBIAN_FRONTEND=noninteractive \
    apt-get update && apt-get install -y --no-install-recommends \
    curl build-essential libssl-dev python3 python3-venv python3-dev python3-pip

# Create a folder, Download the latest runner package, Extract the installer
RUN mkdir -p /home/actions-runner && cd /home/actions-runner \
    && curl -O -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
    && tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

# Install the additional dependencies
RUN /home/actions-runner/bin/installdependencies.sh

COPY start.sh start.sh

# make the script executable
RUN chmod +x start.sh

# set the entrypoint to the start.sh script
ENTRYPOINT ["./start.sh"]