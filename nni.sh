#!/bin/bash
function clean()
{
    rm $flag_file
}
trap 'clean ; exit' HUP INT QUIT TSTP

export PYTHONPATH=$PYTHONPATH:.
NOT_FINISH=1
while [ $NOT_FINISH -ne 0 ]
do
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
    bash -c "$* --auto-resume --cfg-options log_config.hooks.0.log_dir=${NNI_OUTPUT_DIR}/tensorboard" &
    pid=$!

    # create flag file
    flag_file=/home/zh21/${pid}
    if [ ! -f "$flag_file" ]; then
        touch  $flag_file
        chmod 777 $flag_file
    fi

    # monitor flag file every 30 sec
    kill -0 $pid 2>/dev/null
    while [[ -f $flag_file && $? -eq 0 ]]
    do
        sleep 30
        kill -0 $pid 2>/dev/null
    done

    # kill pid
    kill -9 ${pid} 2>/dev/null
    wait $pid 
    NOT_FINISH=$?
done
rm $flag_file