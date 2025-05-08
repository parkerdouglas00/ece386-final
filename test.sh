#!/bin/bash

gpiomon -r -n 1 gpiochip0 105 | while read -r line; do
    echo "You pressed the button!"
    
    sudo docker run --rm \
    --device=/dev/snd \
    --runtime=nvidia \
    --ipc=host \
    -v huggingface:/huggingface/ \
    --add-host=host.docker.internal:host-gateway \
    whisper

done

