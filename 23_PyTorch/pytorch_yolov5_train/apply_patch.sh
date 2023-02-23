#! /bin/bash
if [ -d yolov5 ];then
    echo "yolov5 already exists."
    cd yolov5
    if [ -f "yolov5_mlu.patch" ];then 
        echo "patch file already exists."
    else
        echo 'We will apply yolov5_mlu.patch to origin code.'
        cp ../yolov5_mlu.patch ./
        git apply yolov5_mlu.patch
    fi
else 
    echo 'git clone yolov5, checkout v6.0 and git apply yolov5_mlu.patch.'
    git clone https://github.com/ultralytics/yolov5.git
    cd yolov5
    git checkout v6.0
    cp ../yolov5_mlu.patch ./
    git apply yolov5_mlu.patch
fi
