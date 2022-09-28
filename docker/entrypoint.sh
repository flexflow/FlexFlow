#!/bin/bash

FIRST_BOOT_COMPLETE="first_boot_complete"

if [ ! -f "/usr/$FIRST_BOOT_COMPLETE" ]; then
	echo "Building FlexFlow..."
	cd /usr/FlexFlow && mkdir -p build && cd build && ../config/config.linux && make -j
	echo "Installing FlexFlow..."
    make install
    echo "Installation of FlexFlow successfully completed!"
    touch "/usr/$FIRST_BOOT_COMPLETE"
    cd /usr/FlexFlow
fi

# Now open Bash, the default entrypoint after the first boot
bash
