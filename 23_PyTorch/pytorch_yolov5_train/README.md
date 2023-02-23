# 基于寒武纪 MLU 的模型训练--YOLOv5目标检测
### --PyTorch, Python3, fp32

## 目录
* [实验介绍](#1.实验介绍)
* [实验结构](#2.实验结构)
* [实验讲解](#3.实验讲解)
* [快速开始](#4.快速开始)
* [本地实验步骤](#5本地实验步骤)
  * [下载工程代码](#5.1下载工程代码)
  * [数据集准备](#5.2数据集准备)
  * [安装所需依赖](#5.3安装所需依赖)
  * [移植修改](#5.4移植修改)
  * [训练验证](#5.5训练验证)
* [免责声明](#6.免责声明)
* [Release notes](#7.Release_Notes)

## 1 实验介绍
本实验主要介绍基于寒武纪与寒武纪 Pytorch 框架的YOLOv5（v6.0版本）目标检测训练方法。在官方源码的基础上，需要进行简单移植和修改工作，使 MLU370 加速训练 YOLOv5 算法，后续章节将会详细介绍移植过程。

## 2 实验结构
```   
#本实验目录    

└── pytorch_yolov5_train
    ├── apply_patch.sh           
    ├── course_images
    │   └── yolov5m.png
    ├── pytorch_yolov5_train.ipynb
    ├── README.md
    ├── requirements_mlu.txt
    ├── utils_mlu
    │   ├── collect_env.py
    │   ├── common_utils.sh
    │   ├── configs.json
    │   ├── metric.py
    │   └── __pycache__
    └── yolov5_mlu.patch
```
## 3 实验讲解  
1. 可见pytorch_yolov5_train.ipynb文件，使用jupyter打开，内含具体的操作步骤和解释。

## 4 快速开始  
本实验提供两种快速开始方法：
1. 进入 jupyter notebook 环境进行实验；
2. 克隆本实验工程至本地/寒武纪云平台进行实验，具体步骤如下：

## 5 本地实验步骤  
### 5.1 下载工程代码  
```bash   
git clone https://gitee.com/cambricon/practices.git
```

### 5.2 预训练模型和数据集准备  

**数据集准备**  
1. 数据集下载：http://images.cocodataset.org/zips/val2017.zip, http://images.cocodataset.org/annotations/annotations_trainval2017.zip  
2. 若本地已存在或挂载数据集，需将/your path/datasets/coco(本实验假放置于../../dataset/private/COCO2017/) 软连接到yolov5同级目录。    

**模型准备**  
1. 下载预训练模型：https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt  
2. 根据上述目录结构配置模型路径：./yolov5_model/yolov5m.pt 

```bash 
sh prepare.sh
```

### 5.3 安装所需依赖
```bash 
pip install -r requirements_mlu.txt
```
### 5.4 移植修改

为便于用户移植和修改，我们通过patch方式将适配后的训练代码应用于源码。patch命令如下：

```bash 
bash apply_patch.sh
```
### 5.5 训练验证
#### 训练  

* Train YOLOv5m on COCO for 3 epochs in mlu device  

```bash 
python train.py --imgsz 640 --batch 28 --epochs 3 --data coco.yaml --weights "" --cfg yolov5m.yaml --device mlu  
```
* From pretrained training  

```bash 
python train.py --batch 28 --epochs 1 --data coco.yaml --cfg yolov5m.yaml  --device mlu  --weights yolov5m.pt
```
* Resume Training  

```bash 
python train.py --resume --batch 28 --data coco.yaml --cfg yolov5m.yaml  --device mlu  --weights ./runs/train/exp6/weights/last.pt 
```
参数解析:

* imgsz: 训练、验证时图像大小，默认640；
* epoch: 训练迭代次数；
* data: dataset.yaml 文件路径；
* weights: 初始化权重路径；
* cfg: model.yaml 路径；
* device: 运行设备选取，如mlu，cuda或cpu； 
* resume: 在最近训练结果继续训练；
* 运行命令中可添加 `--pyamp` 参数进行自动混合精度训练；
* 更多参数设置及解析在train.py文件parse_opt函数中查看；
* 超参可在```data/hyps/typ.scratch.yaml```中设置。

#### 精度验证

```bash 
python val.py --data coco.yaml --conf 0.001 --iou 0.65 --weight runs/train/exp/weights/best.pt --device mlu
```
## 6 免责声明
您明确了解并同意，以下链接中的软件、数据或者模型由第三方提供并负责维护。在以下链接中出现的任何第三方的名称、商标、标识、产品或服务并不构成明示或暗示与该第三方或其软件、数据或模型的相关背书、担保或推荐行为。您进一步了解并同意，使用任何第三方软件、数据或者模型，包括您提供的任何信息或个人数据（不论是有意或无意地），应受相关使用条款、许可协议、隐私政策或其他此类协议的约束。因此，使用链接中的软件、数据或者模型可能导致的所有风险将由您自行承担。

* COCO VAL2017 数据集下载链接：http://images.cocodataset.org/zips/val2017.zip

* COCO VAL2017 标签下载链接：http://images.cocodataset.org/annotations/annotations_trainval2017.zip

* YOLOV5M 模型下载链接：https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5m.pt

* YOLOV5 GITHUB下载链接：https://github.com/ultralytics/yolov5.git
## 7 Release_Notes
@TODO
