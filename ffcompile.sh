#!/bin/bash

APP="$1"

if [ -z "$APP" ]; then echo "Usage: ./ffcompile app_dir"; exit; fi

if [ -z "$FF_HOME" ]; then echo "FF_HOME variable is not defined, aborting compile"; exit; fi

if [[ ! -f $FF_HOME/protobuf/src/protoc ]]; then echo "Please build the FlexFlow Protocol Buffer library"; exit; fi

echo "Use the FlexFlow protoc"
$FF_HOME/protobuf/src/protoc -I=$FF_HOME/src/runtime --cpp_out=$FF_HOME/src/runtime $FF_HOME/src/runtime/strategy.proto

cd $APP
make -j 12
