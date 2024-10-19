#!/bin/bash

# Check if CUDA version is supplied
if [ -z "$1" ]; then
  echo "Please provide the CUDA version as XX.Y (e.g., 11.8)"
  exit 1
fi

# Extract major and minor version from input
CUDA_VERSION=$1
MAJOR_VERSION=$(echo "$CUDA_VERSION" | cut -d '.' -f 1)
MINOR_VERSION=$(echo "$CUDA_VERSION" | cut -d '.' -f 2)

# Function to install PyTorch
install_pytorch() {
  local major=$1
  local minor=$2

  echo "Attempting to install PyTorch with CUDA ${major}.${minor} support..."

  # Run dry-run first
  if pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu"${major}${minor}" --dry-run; then
    echo "Dry-run succeeded, proceeding with actual installation..."
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu"${major}${minor}"
    return 0
  else
    echo "Dry-run failed for CUDA ${major}.${minor}."
    return 1
  fi
}

# Try to install with provided CUDA version or lower
while [ "$MINOR_VERSION" -ge 0 ]; do
  if install_pytorch "$MAJOR_VERSION" "$MINOR_VERSION"; then
    echo "PyTorch installation successful with CUDA ${MAJOR_VERSION}.${MINOR_VERSION}"
    exit 0
  else
    # Decrease the minor version
    MINOR_VERSION=$((MINOR_VERSION - 1))

    # Abort if minor version is less than 0 (all <= input failed)
    if [ "$MINOR_VERSION" -lt 0 ]; then
      echo "All minor versions <= input failed. Searching for the smallest minor version."
    fi
  fi
done

# Now attempt to find the smallest available minor version >= 0
MINOR_VERSION=0
echo "Starting search for the smallest minor version..."

while true; do
  if install_pytorch "$MAJOR_VERSION" "$MINOR_VERSION"; then
    echo "PyTorch installation successful with CUDA ${MAJOR_VERSION}.${MINOR_VERSION}"
    exit 0
  else
    # Increase minor version to search for available one
    MINOR_VERSION=$((MINOR_VERSION + 1))

    # Stop if no valid version is found after a certain number of tries
    # For practical purposes, let's assume we won't go beyond minor version 10
    if [ "$MINOR_VERSION" -gt 10 ]; then
      echo "No valid PyTorch installation found for CUDA ${MAJOR_VERSION}. Aborting."
      exit 1
    fi
  fi
done
