import json
import torch
import numpy as np
import os

import magicmind.python.runtime as mm
from magicmind.python.runtime.parser import Parser

batch_size = 1

# 创建 Builder
builder = mm.Builder()

# 创建 MagicMind Network
resnet50_network = mm.Network()

# 创建 MagicMind builder_config
builder_config = mm.BuilderConfig()

# 创建 MagicMind Parser （pytorch后端）
parser = Parser(mm.ModelKind.kPytorch)

# 获取 Network 的输入节点及其维度
parser.set_model_param("pytorch-input-dtypes", [mm.DataType.FLOAT32])

# 将 resnet50_mm.pt 与 MagicMind Network 绑定
parser.parse(resnet50_network, "./resnet50_mm.pt")

# 设置 MagicMind Network 参数：硬件平台、自动int64转int32、卷积折叠，可变输入开关等
build_config = {
	"archs": ["mtp_372"],
	"graph_shape_mutable": True,
	"precision_config": {"precision_mode": "force_float32"},
	"opt_config": {"type64to32_conversion": True, "conv_scale_fold": True}
}
builder_config.parse_from_string(json.dumps(build_config)).ok()

# 执行 MagicMind 模型生成
model = builder.build_model("resnet50_parser.mm", resnet50_network, builder_config)

# 保存 MagicMind 模型至本地
assert model != None
model.serialize_to_file("./resnet50_parser.mm")
print("Generate model done, model save to %s" % "./resnet50_parser.mm")
