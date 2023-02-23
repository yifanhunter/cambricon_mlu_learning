#!/bin/bash

current_dir=$(cd $(dirname $0); pwd)

container_name="cnserving_container"
if [ $# -ne 1 ];then
  echo "Usage: ./start_server.sh image_name"
  echo "       Please specify cnserving image name."
  exit -1
fi
image_name=$1

docker run --name=${container_name} \
       	-t --rm -p 8500:8500 -p 8501:8501 \
        --device=/dev/cambricon_ctl \
        --device=/dev/cambricon_dev0 \
        -v ${current_dir}/sample_add:/models/sample_add \
        -v ${current_dir}/models.config:/models/models.config \
        ${image_name} \
        --model_config_file=/models/models.config \
        &
