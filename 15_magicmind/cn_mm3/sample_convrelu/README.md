# Builder API usage example based on conv and relu

Table Of Contents

- [内容描述](#description)
- [工作原理](#how-does-this-sample-work)
- [编译运行](#compiling-and-running-the-sample)

## 内容描述

sample_convrelu展示了用户如何通过调用MagicMind Python API接口构建卷积+激活网络片段的例子。

## 工作原理

本示例基于MagicMind Python API，展示如何构造卷积+激活网络片段，并进行运行部署前的编译工作，包括：

1. 创建输入tensor。
2. 构建网络：
3. 创建模型。
4. 序列化模型，生成`graph`。

## 运行

执行下列命令：

```bash
python3 sample_convrelu.py

```

该指令会输出文件`graph`，如需运行本示例生成的网络部署文件，可以使用寒武纪对外提供的MagicMind模型部署工具mm_run，其命令如下：

```bash
mm_run --magicmind_model path/to/model  --iterations 1 --duration 0 --warmup 0 --input_dims 4,224,224,3
```

或用户自行根据《寒武纪MagicMind用户手册》Python API章节调用MagicMind运行时。
