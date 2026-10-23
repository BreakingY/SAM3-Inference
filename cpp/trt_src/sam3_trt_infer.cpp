#include <iostream>
#include <filesystem>
#include <fstream>
#include <cuda_runtime.h>
#include "NvInfer.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <cuda_fp16.h>
#include <opencv2/opencv.hpp>
#include <iomanip>
#include <sstream>
#include "sam3_data_process.cuh"
class TRTLogger : public nvinfer1::ILogger
{
    public:
        void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override
    {
        if (severity <= nvinfer1::ILogger::Severity::kVERBOSE)
        {
            std::cout << "\033[1;35m [TRT] " << msg << "\033[0m\n";
        }
    }
};

TRTLogger trt_logger{};


struct Sam3Detection {
    float x1, y1, x2, y2;
    float confidence;
};
static float read_tensor_value(const void* data,nvinfer1::DataType dtype,size_t index)
{
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:
            return static_cast<const float*>(data)[index];

        case nvinfer1::DataType::kHALF:
            return __half2float(
                static_cast<const __half*>(data)[index]);

        default:
            throw std::runtime_error(
                "Unsupported SAM3 output data type");
    }
}

static float sigmoid(float x)
{
    if (x >= 0.0f) {
        return 1.0f / (1.0f + std::exp(-x));
    }
    const float exp_x = std::exp(x);
    return exp_x / (1.0f + exp_x);
}
// pred_boxes:      [batch, num_queries, 4]，候选框坐标 [x1, y1, x2, y2]，归一化坐标
// pred_logits:     [batch, num_queries]，每个候选框的原始置信度 logits
// presence_logits: [batch, 1]，整张图片中目标是否存在的全局 logits
std::vector<Sam3Detection> postprocess_sam3(
    const void* pred_boxes,
    nvinfer1::DataType boxes_dtype,
    const void* pred_logits,
    nvinfer1::DataType logits_dtype,
    const void* presence_logits,
    nvinfer1::DataType presence_dtype,
    int num_queries,
    int img_width,
    int img_height,
    float conf_threshold)
{
    std::vector<Sam3Detection> detections;

    if (!pred_boxes || !pred_logits || !presence_logits ||
        num_queries <= 0 || img_width <= 0 || img_height <= 0) {
        return detections;
    }

    // 场景级目标存在概率
    const float presence_score = sigmoid(read_tensor_value(presence_logits, presence_dtype, 0));

    for (int i = 0; i < num_queries; ++i) {
        // 计算最终置信度
        const float logit = read_tensor_value(pred_logits, logits_dtype, i);

        const float confidence = sigmoid(logit) * presence_score;

        if (confidence < conf_threshold) {
            continue;
        }

        // SAM3 输出归一化 xyxy 坐标
        const size_t offset = static_cast<size_t>(i) * 4;

        float x1 = read_tensor_value(pred_boxes, boxes_dtype, offset);
        float y1 = read_tensor_value(pred_boxes, boxes_dtype, offset + 1);
        float x2 = read_tensor_value(pred_boxes, boxes_dtype, offset + 2);
        float y2 = read_tensor_value(pred_boxes, boxes_dtype, offset + 3);

        // 转换为原图像素坐标
        x1 *= img_width;
        x2 *= img_width;
        y1 *= img_height;
        y2 *= img_height;

        // 限制在图像范围内
        x1 = std::clamp(x1, 0.0f, static_cast<float>(img_width));
        x2 = std::clamp(x2, 0.0f, static_cast<float>(img_width));
        y1 = std::clamp(y1, 0.0f, static_cast<float>(img_height));
        y2 = std::clamp(y2, 0.0f, static_cast<float>(img_height));

        if (x2 <= x1 || y2 <= y1) {
            continue;
        }

        detections.push_back({
            x1, y1, x2, y2, confidence
        });
    }

    return detections;
}

// 将检测结果绘制到图像并保存
void visualize_sam3_results(cv::Mat& img,const std::vector<Sam3Detection>& detections,const std::string& outfile)
{
    for (const auto& det : detections) {
        int x1 = static_cast<int>(std::round(det.x1));
        int y1 = static_cast<int>(std::round(det.y1));
        int x2 = static_cast<int>(std::round(det.x2));
        int y2 = static_cast<int>(std::round(det.y2));

        // 绘制目标框
        cv::rectangle(img,cv::Point(x1, y1),cv::Point(x2, y2),cv::Scalar(0, 255, 0),2);

        // 绘制置信度
        std::ostringstream label;
        label << std::fixed << std::setprecision(3) << det.confidence;

        int baseline = 0;
        cv::Size text_size = cv::getTextSize(label.str(),cv::FONT_HERSHEY_SIMPLEX,0.6,1,&baseline);

        int text_y = std::max(y1, text_size.height + 5);

        cv::rectangle(img,cv::Point(x1, text_y - text_size.height - 5),cv::Point(x1 + text_size.width + 4, text_y + baseline),cv::Scalar(0, 0, 0),cv::FILLED);

        cv::putText(img,label.str(),cv::Point(x1 + 2, text_y - 2),cv::FONT_HERSHEY_SIMPLEX,0.6,cv::Scalar(255, 255, 255),1);
    }

    // 如果输出目录不存在，则创建
    const std::filesystem::path path(outfile);
    if (!path.parent_path().empty()) {
        std::filesystem::create_directories(path.parent_path());
    }

    if (!cv::imwrite(outfile, img)) {
        throw std::runtime_error("Failed to save image: " + outfile);
    }
}
int main(int argc, char* argv[]){
    if (argc < 3){
        std::cout << "Usage: ./bin indir engine_path.engine" << std::endl;
        return 0;
    }
    cudaStream_t sam3_stream;
    std::string in_dir = argv[1];
    const std::string engine_path = argv[2];

    std::unique_ptr<nvinfer1::IRuntime> trt_runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> trt_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> trt_ctx;

    std::vector<std::string> input_names;
    std::vector<std::string> output_names;
    std::vector<void*> input_gpu;
    std::vector<void*> output_gpu;
    std::vector<void*> output_cpu;
    std::vector<size_t> output_sizes;
    int in_width, in_height;
    int img_width = 0, img_height = 0;

    void* input_device = nullptr;
    
    CUDA_CHECK(cudaStreamCreate(&sam3_stream));

    size_t file_size = std::filesystem::file_size(engine_path);
    if (file_size==0){
        std::stringstream err;
        err << "Engine file is empty";
        throw std::runtime_error(err.str());
    }
    char* engine_data = (char *)malloc(file_size);
    std::ifstream file(engine_path, std::ios::binary);
    if (!file.is_open()){
        std::stringstream err;
        err << "File " << engine_path << " could not be opened. Please check permissions\n";
        throw std::runtime_error(err.str());
    }

    file.read(engine_data, file_size);
    file.close();
    trt_runtime = std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(trt_logger));
    trt_engine = std::unique_ptr<nvinfer1::ICudaEngine>(trt_runtime->deserializeCudaEngine(engine_data, file_size));
    free((void*)engine_data);

    trt_ctx = std::unique_ptr<nvinfer1::IExecutionContext>(trt_engine->createExecutionContext());

    int nb_io = trt_engine->getNbIOTensors();
    // ["pixel_values", "input_ids", "attention_mask"]
    // ["pred_boxes", "pred_logits", "presence_logits"]
    int pixel_values_idx, input_ids_idx, attention_mask_idx; 
    int pred_boxes_idx, pred_logits_idx, presence_logits_idx;
    for (int io_idx = 0; io_idx <= nb_io - 1; io_idx++){
        const char* name = trt_engine->getIOTensorName(io_idx);
        nvinfer1::TensorIOMode mode = trt_engine->getTensorIOMode(name);
        auto dtype = trt_engine->getTensorDataType(name);
        std::cout << "IO: " << name << ", dtype: ";
        switch (dtype)
        {
            case nvinfer1::DataType::kFLOAT: std::cout << "float"; break;
            case nvinfer1::DataType::kHALF:  std::cout << "half"; break;
            case nvinfer1::DataType::kINT8:  std::cout << "int8_t"; break;
            case nvinfer1::DataType::kINT32: std::cout << "int32_t"; break;
            case nvinfer1::DataType::kINT64: std::cout << "int64_t"; break;
            case nvinfer1::DataType::kBOOL:  std::cout << "bool"; break;
            default: std::cout << "unknown"; break;
        }
        std::cout << std::endl;

        if (mode == nvinfer1::TensorIOMode::kINPUT){
            std::cout << "Found input tensor " << name << std::endl;
            input_names.push_back(std::string(name));
        }
        else if (mode == nvinfer1::TensorIOMode::kOUTPUT){
            std::cout << "Found output tensor " << name << std::endl;
            output_names.push_back(std::string(name));
        }
        else{
            std::cout << "I/O binding " << name << " has an unknown mode: " << static_cast<int>(mode) << ", this is most likely a bug in the application" << std::endl;
        }

    }
    for(int i = 0; i < input_names.size(); i++){
        std::string name = input_names[i];
        nvinfer1::Dims dims = trt_engine->getTensorShape(name.c_str());
        size_t element_size = 0;
        switch (trt_engine->getTensorDataType(name.c_str())) {
            case nvinfer1::DataType::kFLOAT:
                element_size = sizeof(float);
                break;
            case nvinfer1::DataType::kHALF:
                element_size = sizeof(uint16_t);
                break;
            case nvinfer1::DataType::kINT32:
                element_size = sizeof(int32_t);
                break;
            case nvinfer1::DataType::kINT64:
                element_size = sizeof(int64_t);
                break;
            default:
                throw std::runtime_error("Unsupported output tensor dtype");
        }

        size_t nbytes = element_size;
        for (int idx = 0; idx < dims.nbDims; idx++) {
            nbytes *= dims.d[idx];
        }
        if (name == "pixel_values"){
            in_height= dims.d[2];
            in_width = dims.d[3];
            pixel_values_idx = i;
            std::cout << "Input image has dimensions " << in_width << " x " << in_height << std::endl;
        }
        else if(name == "input_ids"){
            input_ids_idx = i;
        }
        else if(name == "attention_mask"){
            attention_mask_idx = i;
        }
        
        void *gpu_buf;
        CUDA_CHECK(cudaMalloc(&gpu_buf, nbytes));
        cudaMemset(gpu_buf, 0, nbytes);
        input_gpu.push_back(gpu_buf);
        trt_ctx->setTensorAddress(name.c_str(), gpu_buf);
        std::cout << name << "-->batch:" << dims.d[0] << std::endl;
    }
    for(int i = 0; i < output_names.size(); i++){
        std::string name = output_names[i];
        nvinfer1::Dims dims = trt_engine->getTensorShape(name.c_str());
        size_t element_size = 0;
        switch (trt_engine->getTensorDataType(name.c_str())) {
            case nvinfer1::DataType::kFLOAT:
                element_size = sizeof(float);
                break;
            case nvinfer1::DataType::kHALF:
                element_size = sizeof(uint16_t);
                break;
            case nvinfer1::DataType::kINT32:
                element_size = sizeof(int32_t);
                break;
            case nvinfer1::DataType::kINT64:
                element_size = sizeof(int64_t);
                break;
            default:
                throw std::runtime_error("Unsupported output tensor dtype");
        }

        size_t nbytes = element_size;
        for (int idx = 0; idx < dims.nbDims; idx++) {
            nbytes *= dims.d[idx];
        }
        if (name == "pred_boxes"){
            pred_boxes_idx = i;
        }
        else if(name == "pred_logits"){
            pred_logits_idx = i;
        }
        else if(name == "presence_logits"){
            presence_logits_idx = i;
        }
        
        void *cpu_buf, *gpu_buf;

        cpu_buf = malloc(nbytes);
        memset(cpu_buf, 0, nbytes);

        CUDA_CHECK(cudaMalloc(&gpu_buf, nbytes));
        CUDA_CHECK(cudaMemset(gpu_buf, 0, nbytes));
        
        output_gpu.push_back(gpu_buf);
        output_cpu.push_back(cpu_buf);
        output_sizes.push_back(nbytes);
        trt_ctx->setTensorAddress(name.c_str(), gpu_buf);
        std::cout << name << "-->batch:" << dims.d[0] << std::endl;
    }
    // tokenized version of 'person'
    std::vector<int64_t> input_ids={49406,  2533, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407};
    
    std::vector<int64_t> attention_mask={1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0};

    CUDA_CHECK(cudaMemcpyAsync(input_gpu[input_ids_idx], input_ids.data(), input_ids.size() * sizeof(int64_t),  cudaMemcpyHostToDevice, sam3_stream));
    CUDA_CHECK(cudaMemcpyAsync(input_gpu[attention_mask_idx], attention_mask.data(), attention_mask.size() * sizeof(int64_t), cudaMemcpyHostToDevice, sam3_stream));
    CUDA_CHECK(cudaStreamSynchronize(sam3_stream));
    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff;
    float millis_elapsed = 0.0; // int will overflow after ~650 hours
    for (const auto& fname : std::filesystem::directory_iterator(in_dir)){
        if(!std::filesystem::is_regular_file(fname.path())){
            continue;
        }
        std::filesystem::path outfile = std::filesystem::path("results") / fname.path().filename();
        cv::Mat img = cv::imread(fname.path());
        size_t img_size;
        if(img_width != img.cols || img_height != img.rows){
            if (input_device){
                CUDA_CHECK(cudaFree(input_device));
            }
            img_width  = img.cols;
            img_height = img.rows;
            img_size = img_width * img_height * 3;
            if(input_device == nullptr){
                CUDA_CHECK(cudaMalloc(&input_device, img_size));
            }
        }
        start = std::chrono::system_clock::now();
        CUDA_CHECK(cudaMemcpyAsync(input_device, img.data, img_size, cudaMemcpyHostToDevice, sam3_stream));
        data_process_sam3(static_cast<uint8_t*>(input_device), static_cast<float*>(input_gpu[pixel_values_idx]), img_width, img_height, 3, in_width, in_height, sam3_stream);
        bool res = trt_ctx->enqueueV3(sam3_stream);
        if(!res){
            std::cout << "enqueueV3 Err" << std::endl;
        }
        CUDA_CHECK(cudaMemcpyAsync(output_cpu[pred_boxes_idx], output_gpu[pred_boxes_idx], output_sizes[pred_boxes_idx], cudaMemcpyDeviceToHost, sam3_stream));
        CUDA_CHECK(cudaMemcpyAsync(output_cpu[pred_logits_idx], output_gpu[pred_logits_idx], output_sizes[pred_logits_idx], cudaMemcpyDeviceToHost, sam3_stream));
        CUDA_CHECK(cudaMemcpyAsync(output_cpu[presence_logits_idx], output_gpu[presence_logits_idx], output_sizes[presence_logits_idx], cudaMemcpyDeviceToHost, sam3_stream));
        CUDA_CHECK(cudaStreamSynchronize(sam3_stream));
        // 后处理
        int num_queries = trt_engine->getTensorShape(output_names[pred_logits_idx].c_str()).d[1];

        auto detections = postprocess_sam3(output_cpu[pred_boxes_idx], trt_engine->getTensorDataType(output_names[pred_boxes_idx].c_str()),
                                           output_cpu[pred_logits_idx], trt_engine->getTensorDataType(output_names[pred_logits_idx].c_str()),
                                           output_cpu[presence_logits_idx], trt_engine->getTensorDataType(output_names[presence_logits_idx].c_str()),
                                           num_queries,
                                           img_width,
                                           img_height,
                                           0.5f
                                          );

        end = std::chrono::system_clock::now();
        diff = end - start;
        millis_elapsed = (diff.count() * 1000);
        std::cout << "millis_elapsed:" << millis_elapsed << std::endl;
        visualize_sam3_results(img, detections, outfile.string());
    }

    for (auto& ptr : input_gpu){
        if (ptr){
            CUDA_CHECK(cudaFree(ptr));
        }
    }
    for (auto& ptr : output_gpu){
        if (ptr){
            CUDA_CHECK(cudaFree(ptr));
        }
    }
    for (auto& ptr : output_cpu){
        if (ptr){
            free(ptr);
        }
    }
    if (input_device){
        CUDA_CHECK(cudaFree(input_device));
    }
    return 0;
}
