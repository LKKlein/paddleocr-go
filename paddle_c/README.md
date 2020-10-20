# Paddle C预测库目录

## 编译安装
使用cmake编译paddle，并打开-DON_INFER=ON，在编译目录下得到paddle_inference_c_install_dir,将该目录下的所有文件复制到本目录下。

由于官方只提供了C++预测库文件，并未提供C语言预测库API文件，因此需要自己进行编译。详细编译步骤请见https://www.paddlepaddle.org.cn/documentation/docs/zh/advanced_guide/inference_deployment/inference/build_and_install_lib_cn.html#id12