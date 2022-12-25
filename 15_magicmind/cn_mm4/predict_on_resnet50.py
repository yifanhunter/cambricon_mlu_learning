import json
import argparse
import numpy as np

import torch

import magicmind.python.runtime as mm
import magicmind.python.runtime.parser as mm_parser

from torchvision import transforms
from PIL import Image

def parse_option():
    parser = argparse.ArgumentParser('resnet50 script', add_help=False)
    parser.add_argument('--path_mm_model', type=str, metavar="FILE", help='path to model', 
                        default='../cn_mm2/resnet50_from_mm_build.mm')
    args = parser.parse_args()
    return args

def img_preprocess():
    transform = transforms.Compose([
        transforms.Resize(size=292, interpolation=Image.BICUBIC),
        transforms.CenterCrop(size=(256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])

    image_list = ['./ILSVRC2012_val_00000293.JPEG']
    return transform, image_list 


def main(args):
	
    transform, image_list=img_preprocess()

    # 创建 Device
    dev = mm.Device()
    dev.id = 0  #设置 Device Id
    assert dev.active().ok()

    # 加载 MagicMind resnet50 模型
    model = mm.Model()
    model.deserialize_from_file(args.path_mm_model)

    # 创建 MagicMind engine, context
    engine = model.create_i_engine()
    context = engine.create_i_context()

    # 根据 Device 信息创建 queue 实例
    queue = dev.create_queue()
    
    # 准备 MagicMind 输入输出 Tensor 节点
    inputs = context.create_inputs()
    for i in range(len(image_list)):
        path = image_list[i] 
        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        image = transform(img).unsqueeze(0)
        outputs = []
        images = np.float32(image)
        inputs[0].from_numpy(images)
        # 绑定context
        assert context.enqueue(inputs, outputs, queue).ok()
        # 执行推理
        assert queue.sync().ok()
        pred = torch.from_numpy(np.array(outputs[0].asnumpy()))
        
        ## 计算概率，以及输出top1、top5
        pred = pred.squeeze(0)
        pred = torch.softmax(pred, dim=0)
        pred = pred.numpy()
        top_5 = pred.argsort()[-5:][::-1]
        top_1 = pred.argsort()[-1:][::-1]
        print('test img is ', path)
        print('top1_cls:', top_1)
        print('top1_prob:', pred[top_1])
        print('top5_cls:', top_5)
        print('top5_prob:', pred[top_5[0]], pred[top_5[1]], 
               pred[top_5[2]], pred[top_5[3]], pred[top_5[4]])

if __name__ == '__main__':
    args = parse_option()
    main(args)

