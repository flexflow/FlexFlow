#!/bin/bash
set -e

FIRST_BOOT_COMPLETE="first_boot_complete"

if [ ! -f "/usr/$FIRST_BOOT_COMPLETE" ]; then
	echo "Building and installing FlexFlow..."
	cd /usr/FlexFlow && pip install . --verbose
    echo "Installation of FlexFlow completed successfully!"
    touch "/usr/$FIRST_BOOT_COMPLETE"
fi

# Now open Bash, the default entrypoint after the first boot
bash
