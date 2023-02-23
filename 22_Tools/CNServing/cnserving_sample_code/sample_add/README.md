### server端获取镜像

1. 获取docker镜像文件，例如：cnserving-${version}.tar.gz

2. 加载镜像文件

   ```
   docker load -i cnserving-${version}.tar.gz
   ```

### server端模型部署

1. 启动server

   启动server时需确保以下几点：

   1. 将docker内的8500端口映射到docker外
   2. 挂载mlu设备节点
   3. 将模型文件成功映射到docker内，并且model_config_file中配置的base_path与docker内模型路径相同

   ```
   ./start_server.sh cnserving_image_name
   ```

### client发送请求

1. 创建虚拟环境

   ```
   virtualenv client_env --python=python3
   source client_env/bin/activate
   ```

2. 安装requirements

   ```
   pip install -r ../requirements_client.txt
   ```

3. 运行client程序

   ```
   python run_client.py ip:port
   ```
