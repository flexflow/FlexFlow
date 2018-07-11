#!/bin/bash

APP="$1"

if [ -z "$APP" ]; then echo "Usage: ./ffcompile app"; exit; fi

make APP="${APP}"
