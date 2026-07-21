#!/bin/bash

# Runs onnx_export_and_verify.py in a loop with different settings.
#
# Download all weights files to the top level directory before running
# this script.

set -euo pipefail

function log { echo "onnx_verify_many: $@" >&2; }
function error {
	log "error: $1"
	exit 1
}

any_errors=false
for model in yolov3-tiny yolov3 yolov4-tiny yolov4 yolov7-tiny yolov7; do
	for flags in "" "--fp16"; do
		[ -e ../cfg/$model.cfg ] || error "cfg not found for $model"
		[ -e ../$model.weights ] || error "weights not found for $model"

		logfile=$(mktemp --suffix .log)
		log "verifying $model with flags '$flags' (logging to $logfile)"
		if ./onnx_export_and_verify.py $flags --print ../cfg/$model.cfg ../$model.weights ../artwork/*.jpg >$logfile 2>&1; then
			log "...success"
		else
			log "...failed"
			any_errors=true
		fi
	done
done

if $any_errors; then
	error "verification failed"
else
	log "verification succeeded"
fi

