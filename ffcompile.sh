#!/bin/bash

APP="$1"

if [ -z "$APP" ]; then echo "Usage: ./ffcompile app"; exit; fi

make -j 8 APP="${APP}"

