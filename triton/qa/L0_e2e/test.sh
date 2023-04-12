#!/bin/bash
#------------------------------------------------------------------------------#
# Copyright 2022 NVIDIA CORPORATION
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#------------------------------------------------------------------------------#

TEST_PY=operator_test.py
DATADIR="./models"
SERVER=/opt/tritonserver/bin/tritonserver
SERVER_ARGS="--model-repository=$DATADIR"
source ../common/util.sh

rm -f *.log*

RET=0

# 1 GPU 1 node
export REALM_DEFAULT_ARGS="-ll:gpu 1"
TEST_LOG="./single_device_single_node.log"

run_server
if [ "$SERVER_PID" == "0" ]; then
    echo -e "\n***\n*** Failed to start $SERVER\n***"
    cat $SERVER_LOG
    exit 1
fi

set +e
python $TEST_PY >>$TEST_LOG 2>&1
if [ $? -ne 0 ]; then
    echo -e "\n***\n*** Test Failed\n***"
    cat $TEST_LOG
    RET=1
fi
set -e

# [issue #7] WAR to ignore core dump on server exit
set +e
kill_server
set -e

# [gluo FIXME] add test for multi-GPU / multi-node

if [ $RET -eq 0 ]; then
  echo -e "\n***\n*** Test Passed\n***"
else
  echo -e "\n***\n*** Test Failed\n***"
fi

exit $RET
