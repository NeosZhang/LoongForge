#!/bin/bash
set -eo pipefail

SETCOLOR_SUCCESS="echo -en \\E[1;32m"
SETCOLOR_FAILURE="echo -en \\E[1;31m"
SETCOLOR_WARNING="echo -en \\E[1;33m"
SETCOLOR_NORMAL="echo -en  \\E[0;39m"

SUCCESS_echo(){
    $SETCOLOR_SUCCESS && echo "$1" && $SETCOLOR_NORMAL
}

FAILURE_echo(){
    $SETCOLOR_FAILURE && echo "$1" && $SETCOLOR_NORMAL
}

WARNING_echo(){
    $SETCOLOR_WARNING && echo "$1" && $SETCOLOR_NORMAL
}

SUCCESS_echo_date(){
    $SETCOLOR_SUCCESS && echo "$(date "+%Y-%m-%d_%H:%M.%S"): $1" && $SETCOLOR_NORMAL
}

FAILURE_echo_date(){
    $SETCOLOR_FAILURE && echo "$(date "+%Y-%m-%d_%H:%M.%S"): $1" && $SETCOLOR_NORMAL
}

WARNING_echo_date(){
    $SETCOLOR_WARNING && echo "$(date "+%Y-%m-%d_%H:%M.%S"): $1" && $SETCOLOR_NORMAL
}
