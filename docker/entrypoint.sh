#!/bin/bash

FIRST_BOOT_COMPLETE="first_boot_complete"

if [ ! -f "/usr/$FIRST_BOOT_COMPLETE" ]; then
	echo "Building FlexFlow (C++)..."
	cd /usr/FlexFlow && mkdir build && cd build && ../config/config.linux && make -j
	echo "Installing FlexFlow (C++)..."
    cd /usr/FlexFlow/build && make install
    echo "Installation of FlexFlow (C++) successfully completed!"
    echo "Installing FlexFlow (Python)..."
    cd /usr/FlexFlow && pip install .
    echo "Installation of FlexFlow (Python) successfully completed!"
    touch "/usr/$FIRST_BOOT_COMPLETE"
fi

# Now open Bash, the default entrypoint after the first boot
bash
