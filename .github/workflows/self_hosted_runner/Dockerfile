# Acknowledged to https://testdriven.io/blog/github-actions-docker/
FROM ubuntu:20.04

# set the github runner version
ARG RUNNER_VERSION="2.298.2"

# update the base packages and add a non-sudo user
RUN apt-get update -y && apt-get upgrade -y && useradd -m docker

# install python and the packages the your code depends on along with jq so we can parse JSON
# add additional packages as necessary
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    sudo curl jq build-essential libxml2 libssl-dev libffi-dev python3 python3-venv python3-dev python3-pip

RUN adduser docker sudo && echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
RUN chown -R docker /opt
# cd into the user directory, download and unzip the github actions runner
RUN cd /home/docker && mkdir actions-runner && cd actions-runner \
    && curl -O -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz \
    && tar xzf ./actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz

# install some additional dependencies
RUN chown -R docker ~docker && DEBIAN_FRONTEND=noninteractive /home/docker/actions-runner/bin/installdependencies.sh

# copy over the start.sh script
COPY start.sh start.sh

# make the script executable
RUN chmod +x start.sh

# since the config and run script for actions are not allowed to be run by root,
# set the user to "docker" so all subsequent commands are run as the docker user
USER docker

# set the entrypoint to the start.sh script
ENTRYPOINT ["./start.sh"]
