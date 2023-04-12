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

SERVER_LOG=${SERVER_LOG:=./server.log}
SERVER_TIMEOUT=${SERVER_TIMEOUT:=60}
SERVER_LD_PRELOAD=${SERVER_LD_PRELOAD:=""}

# Run inference server. Return once server's health endpoint shows
# ready or timeout expires. Sets SERVER_PID to pid of SERVER, or 0 if
# error (including expired timeout)
function run_server () {
    SERVER_PID=0

    if [ -z "$SERVER" ]; then
        echo "=== SERVER must be defined"
        return
    fi

    if [ ! -f "$SERVER" ]; then
        echo "=== $SERVER does not exist"
        return
    fi

    if [ -z "$SERVER_LD_PRELOAD" ]; then
      echo "=== Running $SERVER $SERVER_ARGS"
    else
      echo "=== Running LD_PRELOAD=$SERVER_LD_PRELOAD $SERVER $SERVER_ARGS"
    fi

    LD_PRELOAD=$SERVER_LD_PRELOAD $SERVER $SERVER_ARGS > $SERVER_LOG 2>&1 &
    SERVER_PID=$!

    wait_for_server_ready $SERVER_PID $SERVER_TIMEOUT
    if [ "$WAIT_RET" != "0" ]; then
        kill $SERVER_PID || true
        SERVER_PID=0
    fi
}

# Wait until server health endpoint shows ready. Sets WAIT_RET to 0 on
# success, 1 on failure
function wait_for_server_ready() {
    local spid="$1"; shift
    local wait_time_secs="${1:-30}"; shift

    WAIT_RET=0

    local wait_secs=$wait_time_secs
    until test $wait_secs -eq 0 ; do
        if ! kill -0 $spid; then
            echo "=== Server not running."
            WAIT_RET=1
            return
        fi

        sleep 1;

        set +e
        code=`curl -s -w %{http_code} localhost:8000/v2/health/ready`
        set -e
        if [ "$code" == "200" ]; then
            return
        fi

        ((wait_secs--));
    done

    echo "=== Timeout $wait_time_secs secs. Server not ready."
    WAIT_RET=1
}

# Kill inference server. SERVER_PID must be set to the server's pid.
function kill_server () {
    kill $SERVER_PID
    wait $SERVER_PID
}
