#!/bin/bash
export PYTHONPATH=$PYTHONPATH:.
export CUDA_VISIBLE_DEVICES=$(
echo $"
from os import popen
from time import sleep
from random import randint
cmd = \"nvidia-smi | egrep -o '[0-9]*MiB /' | egrep -o '[0-9]*' | xargs echo\"
delay = ${DELAY:=4}
interval=randint(1,10)
while delay:
    sleep(interval)
    interval = interval * 2 if interval < 180 else interval
    gpu_available = [
        str(i) for i,mem in enumerate(popen(cmd).read().strip().split())
        if int(mem) < ${IDLE_THR:-100}
    ]
    if len(gpu_available) >= ${GPU_NEED:=1}:
        delay -= 1
    else:
        delay = ${DELAY}
print(','.join(gpu_available[:${GPU_NEED}]))
" |  python - 
)
echo CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}
echo $*
bash -c "$* --cfg-options log_config.hooks.0.log_dir=${NNI_OUTPUT_DIR}/tensorboard"  