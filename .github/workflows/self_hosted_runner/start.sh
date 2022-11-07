#!/bin/bash
set -x
ORGANIZATION=$ORGANIZATION
REGISTRATION_TOKEN=$REGISTRATION_TOKEN

cd /home/docker/actions-runner

./config.sh --url https://github.com/${ORGANIZATION} --token ${REGISTRATION_TOKEN}

cleanup() {
    echo "Removing runner..."
    ./config.sh remove --token ${REGISTRATION_TOKEN}
}

trap 'cleanup; exit 130' INT
trap 'cleanup; exit 143' TERM

./run.sh & wait $!