#!/bin/bash

FIRST_BOOT_COMPLETE="first_boot_complete"

if [ ! -f "/usr/$FIRST_BOOT_COMPLETE" ]; then
	cores_available=$(nproc --all)
	n_build_cores=$(( cores_available -1 )) 
	echo "Building FlexFlow..."
	cd /usr/FlexFlow && mkdir -p build && cd build && ../config/configure.sh && make -j $n_build_cores
	echo "Installing FlexFlow..."
    make install
    echo "Installation of FlexFlow successfully completed!"
    touch "/usr/$FIRST_BOOT_COMPLETE"
    cd /usr/FlexFlow
fi

# Now open Bash, the default entrypoint after the first boot
bash
