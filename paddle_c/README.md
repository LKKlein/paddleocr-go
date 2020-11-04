# Paddle C预测库头文件

## 编译安装
使用cmake编译paddle，并打开-DON_INFER=ON，在编译目录下得到fluid_inference_c_install_dir,将该目录下的paddle/include头文件复制到此处。并将该目录下的paddle/lib配置到动态库环境变量。

详细编译步骤请参见[README.md](../README.md) 或者官方文档指导 https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html#id12