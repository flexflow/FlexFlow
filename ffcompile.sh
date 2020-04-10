#!/bin/bash

APP="$1"

if [ -z "$APP" ]; then echo "Usage: ./ffcompile app_dir"; exit; fi

if [ -z "$FF_HOME" ]; then echo "FF_HOME variable is not defined, aborting compile"; exit; fi

if hash protoc 2>/dev/null; then
    echo "Use the system protoc"
    protoc -I=$FF_HOME/src/runtime --cpp_out=$FF_HOME/src/runtime $FF_HOME/src/runtime/strategy.proto
else
    echo "Use the FlexFlow protoc"
    $FF_HOME/protobuf/src/protoc -I=$FF_HOME/src/runtime --cpp_out=$FF_HOME/src/runtime $FF_HOME/src/runtime/strategy.proto
fi

cd $APP
make -j 12
