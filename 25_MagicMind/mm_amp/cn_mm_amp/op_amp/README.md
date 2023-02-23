## Build the project
```bash
mkdir build
cd build
cmake .. -DABI=0
make
```

## Run demo
```bash
./baidu_test model
```

## 说明：
使用c++ api搭建了一个简单网络，conv+relu+conv+relu，通过代码中宏#define Setfp16conv控制是否对第二个conv单独编译为fp16的conv。  
生成的模型ir图可以在生成模型后，build/compile_graph/process_main_process_legalize1/0_1_SymbolReturn位置看到第二个conv在编译期设置为fp16。
