#!/bin/bash
set -euo pipefail

cd src/runtime
protoc --cpp_out=. strategy.proto
cd ../..
