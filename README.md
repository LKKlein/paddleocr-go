# PaddleOCR-GO/dev_cxx

本服务是[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)的golang部署版本。
dev_cxx分支是直接调用PaddlePaddle的C++推理库，无需自己编译C语言推理库。

## 使用方式

1. 根据个人情况下载对应版本的C++推理库

这里以其中一个版本为例，具体版本详见 https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html#id1

```shell
wget https://paddle-inference-lib.bj.bcebos.com/1.8.4-gpu-cuda9-cudnn7-avx-mkl/fluid_inference.tgz -O fluid_inference.tgz
tar -xzvf fluid_inference.tgz

# 将推理库移动到本目录下的paddle_cxx文件夹
mv fluid_inference/fluid_inference_install_dir/* paddle_cxx/
rm -rf fluid_inference fluid_inference.tgz
```

2. 直接执行demo编译，执行

```shell
go build demo.go

./demo --image images/test.jpg
```