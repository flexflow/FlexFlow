#!/bin/bash
set -euo pipefail
set -x

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Install CUDNN
python_version=${1:-latest}
echo "current python version is ${python_version}"
echo "downloading python from miniconda"
PYTHON_LINK=https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh
PYTHON_TARBALL_NAME=Miniconda3-latest-Linux-x86_64.sh
if [[ "$python_version" == "3.8" ]]; then
    PYTHON_LINK=https://repo.continuum.io/miniconda/Miniconda3-py38_23.5.2-0-Linux-x86_64.sh
    PYTHON_TARBALL_NAME=Miniconda3-py38_23.5.2-0-Linux-x86_64.sh
elif [[ "$python_version" == "3.9" ]]; then
    PYTHON_LINK=https://repo.continuum.io/miniconda/Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
    PYTHON_TARBALL_NAME=Miniconda3-py39_23.5.2-0-Linux-x86_64.sh
elif [[ "$python_version" == "3.10" ]]; then
    PYTHON_LINK=https://repo.continuum.io/miniconda/Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
    PYTHON_TARBALL_NAME=Miniconda3-py310_23.5.2-0-Linux-x86_64.sh
elif [[ "$python_version" == "3.11" ]]; then
    PYTHON_LINK=https://repo.continuum.io/miniconda/Miniconda3-py311_23.5.2-0-Linux-x86_64.sh
    PYTHON_TARBALL_NAME=Miniconda3-py311_23.5.2-0-Linux-x86_64.sh
fi

wget -c -q $PYTHON_LINK && \
    mv $PYTHON_TARBALL_NAME ~/$PYTHON_TARBALL_NAME && \
    chmod +x ~/$PYTHON_TARBALL_NAME && \
    bash ~/$PYTHON_TARBALL_NAME -b -p /opt/conda && \
    rm ~/$PYTHON_TARBALL_NAME && \
    /opt/conda/bin/conda upgrade --all && \
    /opt/conda/bin/conda install conda-build conda-verify && \
    /opt/conda/bin/conda clean -ya