# 说明
* 基于`https://github.com/dataplayer12/SAM3-TensorRT`进行修改，修改如下：
  * 加载本地模型
  * 修改onnx输出, ["instance_masks", "semantic_seg"]-->["pred_boxes", "pred_logits", "presence_logits"] (输出目标bbox和置信度)
  * 新增simplify_tokenizer.py，把提示词转成input_ids和attention_mask(SAM3推理输入)
  * 支持TensorRT推理(NVIDIA)、CANN推理(晟腾 NPU)加速

# 路径设置
* export SAM3_PATH=YOUR_PATH

# 模型下载
* modelscope download --model facebook/sam3 --local_dir ${SAM3_PATH}/model_sam3
* 项目未用到sam3.1

# 导出onnx(NVIDIA)
1. 准备模型
  * cp -r ${SAM3_PATH}/model_sam3 ${SAM3_PATH}/SAM3-Inference/python/
2. 准备导出onnx环境
  * cd ${SAM3_PATH}/SAM3-Inference
  * x86: docker build -t sam3-onnx -f docker/Dockerfile.x86 .
  * arm: docker build -t sam3-onnx-aarch64 -f docker/Dockerfile.aarch64 .
  * 创建容器(x86为例)：docker run -itd --rm  --network=host --gpus "device=2"  --name sam3_onnx --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 --runtime=nvidia -v  ${SAM3_PATH}/SAM3-Inference:/workspace -w /workspace sam3-onnx:latest bash
  * docker exec -it sam3_onnx bash
  * cd python
  * 导出onnx: python onnxexport.py

# TensorRT推理
1. 转模型
  * x86
    * docker pull nvcr.io/nvidia/tensorrt:25.10-py3 # TensorRT 10.13.3 + cuda13
    * docker run -itd --rm --gpus "device=3" --name sam3_trt_infer -v ${SAM3_PATH}/soft:/soft -v ${SAM3_PATH}/SAM3-Inference:/workspace -w /workspace  nvcr.io/nvidia/tensorrt:25.10-py3
    * docker exec -it sam3_trt_infer bash
    * cd python
  * arm(Jetson)
    * JetPack6.x
    * 把生成的的onnx_weights直接拷贝到SAM3-Inference/python下面
    * cd ${SAM3_PATH}/SAM3-Inference/python
  * trtexec --onnx=onnx_weights/sam3_dynamic.onnx --saveEngine=sam3_fp16.plan --fp16 --verbose
2. 安装opencv
  * x86: docker exec -it sam3_trt_infer bash && cd /soft
  * apt update
  ```
  apt install -y \
      build-essential \
      cmake \
      git \
      pkg-config \
      libgtk2.0-dev \
      libgtk-3-dev \
      libavcodec-dev \
      libavformat-dev \
      libswscale-dev \
      libv4l-dev \
      libxvidcore-dev \
      libx264-dev \
      libjpeg-dev \
      libpng-dev \
      libtiff-dev \
      gfortran \
      openexr \
      libatlas-base-dev \
      python3-dev \
      python3-numpy
  ```
  * wget https://github.com/opencv/opencv/archive/4.12.0.tar.gz -O opencv-4.12.0.tar.gz
  * tar -xf opencv-4.12.0.tar.gz
  * cd opencv-4.12.0
  * mkdir build
  * cd build
  ```
  cmake \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr/local \
      -DBUILD_TESTS=OFF \
      -DBUILD_PERF_TESTS=OFF \
      -DOPENCV_GENERATE_PKGCONFIG=YES \
      -DBUILD_EXAMPLES=OFF \
      ..
  ```
  * make -j
  * make install
  * 验证: pkg-config --modversion opencv4
3. 推理
  * x86: docker exec -it sam3_trt_infer bash && cd cpp/trt_src
  * arm(Jetson): cd ${SAM3_PATH}/SAM3-Inference/cpp/trt_src 
  * mkdir build && cd build
  * cmake ..
  * make
  * ./sam3_pcs_app ../../../media/0001/ ../../../python/sam3_fp16.plan

# CANN推理(TODO)
* 把生成的的onnx_weights直接拷贝到SAM3-Inference/python下面
