
MODEL="../../model/pretrained/pytorch_yolov5/" 
DATA="../../dataset/private/COCO2017/"


#1  pretrained model


if [ -f "${MODEL}yolov5m.pt" ] && [ ! -f "./yolov5_model/yolov5m.pt" ] 
then 
    cp ../../model/pretrained/pytorch_yolov5/yolov5m.pt ./yolov5_model/
    echo "The pretrained_model is ready"
elif [ -f "./yolov5_model/yolov5m.pt" ]  
then 
    echo "The pretrained_model exists"
else
    wget -c "https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt" -O "./yolov5_model/"
    echo "The models download to './yolov5_model/' "
fi

# 2 datasets
if [ -d "${DATA}labels" ] &&  [ -d "${DATA}images" ] && [ -f "${DATA}train2017.txt" ] && [ -f "${DATA}val2017.txt" ] && [ ! -d "./datasets/coco/images" ] 
then
# Default data path: :/workspace/dataset/private/COCO2017
    if [ ! -d "./datasets/coco/" ]
    then
        mkdir -p ./datasets/coco/
    fi
    ln -s /workspace/dataset/private/COCO2017/* ./datasets/coco/
    echo "The datasets is ready"
elif [ -d "./datasets/coco/images" ] && [ -d "./datasets/coco/labels" ] && [ -f "./datasets/coco/train2017.txt" ] && [ -f "./datasets/coco/val2017.txt" ] 
then 
    echo "The datasets exists"
else
    echo "Loading datasets"
    !cd yolov5 &&  bash ./data/scripts/get_coco.sh
    echo "The data was downloaded successfully"
fi