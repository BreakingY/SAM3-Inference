#include <iostream>
#include <string>
#include <vector>
#include <iomanip>
#include <sstream>
#include <filesystem>
#include <fstream>
#include <acl.h>
#include <acl_rt.h>
#include <hi_dvpp.h>
#include <ops/acl_dvpp.h>
#include <opencv2/opencv.hpp>
#define SAM3_IMG_MEAN 0.5
#define SAM3_IMG_STD 0.5
// #define AIPP_DYNAMIC
#ifndef CHECK_ACL
#define CHECK_ACL(ret) \
    do { \
        if ((ret) != ACL_SUCCESS) { \
            fprintf(stderr, "Error: ACL returned %d in file %s at line %d\n", \
                    (ret), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
#endif

#ifndef CHECK_DVPP_MPI
#define CHECK_DVPP_MPI(ret) \
    do { \
        if ((ret) != HI_SUCCESS) { \
            fprintf(stderr, "Error: ACL DVPP MPI returned %0x in file %s at line %d\n", \
                    (ret), __FILE__, __LINE__); \
            exit(1); \
        } \
    } while (0)
#endif
#define ALIGN_UP(num, align) (((num) + (align) - 1) & ~((align) - 1))
#define ALIGN_UP2(num) ALIGN_UP(num, 2)
#define ALIGN_UP16(num) ALIGN_UP(num, 16)
#define ALIGN_UP128(num) ALIGN_UP(num, 128)
std::string GetType(aclDataType type){
    std::string type_str;
    switch (type)
    {
    case ACL_DT_UNDEFINED:
        type_str = std::string("ACL_DT_UNDEFINED");
        break;
    case ACL_FLOAT:
        type_str = std::string("ACL_FLOAT");
        break;
    case ACL_FLOAT16:
        type_str = std::string("ACL_FLOAT16");
        break;
    case ACL_INT8:
        type_str = std::string("ACL_INT8");
        break;
    case ACL_INT32:
        type_str = std::string("ACL_INT32");
        break;
    case ACL_UINT8:
        type_str = std::string("ACL_UINT8");
        break;
    case ACL_INT16:
        type_str = std::string("ACL_INT16");
        break;
    case ACL_UINT16:
        type_str = std::string("ACL_UINT16");
        break;
    case ACL_UINT32:
        type_str = std::string("ACL_UINT32");
        break;
    case ACL_INT64:
        type_str = std::string("ACL_INT64");
        break;
    case ACL_UINT64:
        type_str = std::string("ACL_UINT64");
        break;
    case ACL_DOUBLE:
        type_str = std::string("ACL_DOUBLE");
        break;
    case ACL_BOOL:
        type_str = std::string("ACL_BOOL");
        break;
    case ACL_STRING:
        type_str = std::string("ACL_STRING");
        break;
    case ACL_COMPLEX64 :
        type_str = std::string("ACL_COMPLEX64");
        break;
    case ACL_COMPLEX128:
        type_str = std::string("ACL_COMPLEX128");
        break;
    case ACL_BF16:
        type_str = std::string("ACL_BF16");
        break;
    case ACL_INT4:
        type_str = std::string("ACL_INT4");
        break;
    case ACL_UINT1:
        type_str = std::string("ACL_UINT1");
        break;
    case ACL_COMPLEX32:
        type_str = std::string("ACL_COMPLEX32");
        break;
    case ACL_HIFLOAT8:
        type_str = std::string("ACL_HIFLOAT8");
        break;
    case ACL_FLOAT8_E5M2:
        type_str = std::string("ACL_FLOAT8_E5M2");
        break;
    case ACL_FLOAT8_E4M3FN:
        type_str = std::string("ACL_FLOAT8_E4M3FN");
        break;
    case ACL_FLOAT8_E8M0:
        type_str = std::string("ACL_FLOAT8_E8M0");
        break;
    case ACL_FLOAT6_E3M2:
        type_str = std::string("ACL_FLOAT6_E3M2");
        break;
    case ACL_FLOAT6_E2M3:
        type_str = std::string("ACL_FLOAT6_E2M3");
        break;
    case ACL_FLOAT4_E2M1:
        type_str = std::string("ACL_FLOAT4_E2M1");
        break;
    case ACL_FLOAT4_E1M2:
        type_str = std::string("ACL_FLOAT4_E1M2");
        break;
    default:
        break;
    }
    return type_str;
}
void data_process_sam3(uint8_t* src, uint8_t* dst, int src_width, int src_height, int src_channels, int dst_width, int dst_height, hi_vpc_chn channel_id_resize, aclrtStream &stream){
    int ret = 0;

    hi_vpc_pic_info input_pic;
    input_pic.picture_width = src_width;
    input_pic.picture_height = src_height;
    input_pic.picture_format = HI_PIXEL_FORMAT_BGR_888;
    input_pic.picture_width_stride = src_width * 3;
    input_pic.picture_height_stride = src_height;
    input_pic.picture_buffer_size = src_height * src_width * 3;
    input_pic.picture_address = src;

    hi_vpc_pic_info output_pic;
    output_pic.picture_width = dst_width;
    output_pic.picture_height = dst_height;
    output_pic.picture_format = HI_PIXEL_FORMAT_BGR_888;
    output_pic.picture_width_stride = dst_width * 3;
    output_pic.picture_height_stride = dst_height;
    output_pic.picture_buffer_size = dst_height * dst_width * 3;
    output_pic.picture_address = dst;

    uint32_t task_id = 0;
    hi_double fx = (double)dst_width / src_width;
    hi_double fy = (double)dst_height / src_height;
    CHECK_DVPP_MPI(hi_mpi_vpc_resize(channel_id_resize, &input_pic, &output_pic, fx, fy, 0, &task_id, -1));
    CHECK_DVPP_MPI(hi_mpi_vpc_get_process_result(channel_id_resize, task_id, -1));
    return;
}
struct Sam3Detection {
    float x1, y1, x2, y2;
    float confidence;
};
static float read_tensor_value(const void* data,size_t index)
{
    return static_cast<const float*>(data)[index];
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
    const void* pred_logits,
    const void* presence_logits,
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
    const float presence_score = sigmoid(read_tensor_value(presence_logits, 0));

    for (int i = 0; i < num_queries; ++i) {
        // 计算最终置信度
        const float logit = read_tensor_value(pred_logits, i);

        const float confidence = sigmoid(logit) * presence_score;

        if (confidence < conf_threshold) {
            continue;
        }

        // SAM3 输出归一化 xyxy 坐标
        const size_t offset = static_cast<size_t>(i) * 4;

        float x1 = read_tensor_value(pred_boxes, offset);
        float y1 = read_tensor_value(pred_boxes, offset + 1);
        float x2 = read_tensor_value(pred_boxes, offset + 2);
        float y2 = read_tensor_value(pred_boxes, offset + 3);

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
    std::string in_dir = argv[1];
    const std::string engine_path = argv[2];
    int device_id = 0;
    uint32_t model_id;
    aclrtStream stream;
    hi_vpc_chn channel_id_resize;

    aclmdlDesc *model_desc = nullptr;
    aclmdlDataset *input = nullptr;
    aclmdlDataset *output = nullptr;
    aclDataBuffer* data_buf_inputs[3] = {nullptr, nullptr, nullptr};
    std::vector<aclDataBuffer*> output_data_buf;
    
    uint32_t input_num, output_num;
    std::vector<void*> output_buf;
    std::vector<void*> output_cpu;
    std::vector<size_t> output_size;
    
    int batch_size = 1;
    int num_queries = 200;
#ifdef AIPP_DYNAMIC
    size_t aipp_index;
    void *input_AIPP = nullptr;
#endif

    int input_h;
    int input_w;
    void *input_addr_pixel_values;
    void *input_addr_input_ids;
    void *input_addr_attention_mask;
    
    int img_width = 0, img_height = 0;
    void* input_device = nullptr;
    
    int pixel_values_idx, input_ids_idx, attention_mask_idx;
#ifdef AIPP_DYNAMIC
    int aipp_data_idx; 
#endif
    int pred_boxes_idx, pred_logits_idx, presence_logits_idx;

    CHECK_ACL(aclInit(nullptr));
    CHECK_ACL(aclrtSetDevice(device_id));
    CHECK_DVPP_MPI(hi_mpi_sys_init()); // hi_mpi_sys_init 必须在aclrtSetDevice之后
    CHECK_ACL(aclmdlLoadFromFile(engine_path.c_str(), &model_id));
    model_desc = aclmdlCreateDesc();
    if(model_desc == nullptr){
        std::cout << "aclmdlCreateDesc error" << std::endl;
        return -1;
    }
    CHECK_ACL(aclmdlGetDesc(model_desc, model_id));
    input = aclmdlCreateDataset();
    if (input == nullptr) {
        std::cout << "aclmdlCreateDataset input error" << std::endl;
        return -1;
    }
    output = aclmdlCreateDataset();
    if (output == nullptr) {
        std::cout << "aclmdlCreateDataset ouput error" << std::endl;
        return -1;
    }
    /*
    intput name: pixel_values type: ACL_UINT8
    intput name: input_ids type: ACL_INT64
    intput name: attention_mask type: ACL_INT64
    #ifdef AIPP_DYNAMIC
    intput name: ascend_dynamic_aipp_data type: ACL_UINT8
    #endif
    output name: PartitionedCall_/sam3/Gather_2_gatherv2_1731:0:pred_boxes type: ACL_FLOAT
    output name: PartitionedCall_/sam3/Gather_1_gatherv2_1123:0:pred_logits type: ACL_FLOAT
    output name: PartitionedCall_/sam3/Gather_3_gatherv2_1735:0:presence_logits type: ACL_FLOAT
    */
#ifdef AIPP_DYNAMIC
    CHECK_ACL(aclmdlGetInputIndexByName(model_desc, ACL_DYNAMIC_AIPP_NAME, &aipp_index));
    std::cout << "AIPP input index: " << aipp_index << std::endl;

    size_t data_len_aipp = aclmdlGetInputSizeByIndex(model_desc, aipp_index);
    CHECK_ACL(aclrtMalloc(&input_AIPP, data_len_aipp, ACL_MEM_MALLOC_NORMAL_ONLY));
#endif

    aclmdlIODims input_dims_pixel_values;
    CHECK_ACL(aclmdlGetInputDims(model_desc, 0, &input_dims_pixel_values));
    // 格式不是NCHW，而是NHWC(因为aipp的输入格式为RGB888_U8)
    input_dims_pixel_values.dims[1] > 0 ? input_h = input_dims_pixel_values.dims[1]: input_h = 1008;
    input_dims_pixel_values.dims[2] > 0 ? input_w = input_dims_pixel_values.dims[2]: input_w = 1008;
    int channel = input_dims_pixel_values.dims[3];
    std::cout << "batch: " << input_dims_pixel_values.dims[0] << " input_h: " << input_h << " input_w: " << input_w << " channel: " << channel << std::endl;

    CHECK_DVPP_MPI(hi_mpi_dvpp_malloc(0, &input_addr_pixel_values, batch_size * input_h * input_w * 3));
    CHECK_DVPP_MPI(hi_mpi_dvpp_malloc(0, &input_addr_input_ids, batch_size * 32 * sizeof(int64_t)));
    CHECK_DVPP_MPI(hi_mpi_dvpp_malloc(0, &input_addr_attention_mask, batch_size * 32 * sizeof(int64_t)));
    CHECK_ACL(aclrtCreateStream(&stream));

    input_num = aclmdlGetNumInputs(model_desc);
    for (size_t i = 0; i < input_num; ++i) {
        size_t buf_size = aclmdlGetInputSizeByIndex(model_desc, i);
        std::string name = aclmdlGetInputNameByIndex(model_desc, i);
        aclDataBuffer* data_buf;
        if (name.find("pixel_values") != std::string::npos){
            data_buf = aclCreateDataBuffer(nullptr,0);
            if (data_buf == nullptr) {
                std::cout << "aclCreateDataBuffer input error" << std::endl;
                return -1;
            }
            CHECK_ACL(aclmdlAddDatasetBuffer(input, data_buf));
            pixel_values_idx = i;
            data_buf_inputs[pixel_values_idx] = data_buf;
            buf_size = batch_size * input_h * input_w * 3;
        }
        else if(name.find("input_ids") != std::string::npos){
            data_buf = aclCreateDataBuffer(nullptr,0);
            if (data_buf == nullptr) {
                std::cout << "aclCreateDataBuffer input error" << std::endl;
                return -1;
            }
            CHECK_ACL(aclmdlAddDatasetBuffer(input, data_buf));
            input_ids_idx = i;
            data_buf_inputs[input_ids_idx] = data_buf;
            buf_size = batch_size * 32 * sizeof(int64_t);
        }
        else if(name.find("attention_mask") != std::string::npos){
            data_buf = aclCreateDataBuffer(nullptr,0);
            if (data_buf == nullptr) {
                std::cout << "aclCreateDataBuffer input error" << std::endl;
                return -1;
            }
            CHECK_ACL(aclmdlAddDatasetBuffer(input, data_buf));
            attention_mask_idx = i;
            data_buf_inputs[attention_mask_idx] = data_buf;
            buf_size = batch_size * 32 * sizeof(int64_t);
        }
#ifdef AIPP_DYNAMIC
        else if(name.find("aipp_data") != std::string::npos){
            aipp_data_idx = i;
            assert(aipp_data_idx == aipp_index);
            data_buf = aclCreateDataBuffer(input_AIPP, buf_size);
            if (data_buf == nullptr) {
                std::cout << "aclCreateDataBuffer input error" << std::endl;
                return -1;
            }
            CHECK_ACL(aclmdlAddDatasetBuffer(input, data_buf));
            aclmdlAIPP* aipp_param_tensor = aclmdlCreateAIPP(batch_size);
            if(aipp_param_tensor == nullptr){
                std::cout << "aclmdlCreateAIPP error" << std::endl;
                return -1;
            }
            CHECK_ACL(aclmdlSetAIPPInputFormat(aipp_param_tensor, ACL_RGB888_U8)); // 设置输入格式, 
            CHECK_ACL(aclmdlSetAIPPRbuvSwapSwitch(aipp_param_tensor, 1)); // 通道（R/B 、U/V）交换开关 0-不交换 1-交换 输入为BGR，所以需要交换R/B通道
            CHECK_ACL(aclmdlSetAIPPSrcImageSize(aipp_param_tensor, input_w, input_h)); // 输入图像尺寸
            /*
            设置 AIPP（图像预处理引擎）中的缩放参数 
            int8_t scfSwitch: 缩放开关。非零值-启用缩放；0-禁用缩放。
            int32_t scfInputSizeW: 输入图像的宽度，用于缩放计算。
            int32_t scfInputSizeH: 输入图像的高度，用于缩放计算。
            int32_t scfOutputSizeW: 输出图像的目标宽度。
            int32_t scfOutputSizeH: 输出图像的目标高度。
            uint64_t batchIndex: 批处理参数的索引，通常用于处理批量图像。
            */
            CHECK_ACL(aclmdlSetAIPPScfParams(aipp_param_tensor, 0, 0, 0, 0, 0, 0));
            /*
            设置 AIPP（图像预处理引擎）中的裁剪参数
            int8_t cropSwitch: 裁剪开关。非零值-启用裁剪 0-禁用裁剪。
            int32_t cropStartPosW: 裁剪区域的起始水平位置（X 坐标）。
            int32_t cropStartPosH: 裁剪区域的起始垂直位置（Y 坐标）。
            int32_t cropSizeW: 裁剪区域的宽度。
            int32_t cropSizeH: 裁剪区域的高度。
            uint64_t batchIndex: 批处理参数的索引，通常用于处理批量图像。
            */
            CHECK_ACL(aclmdlSetAIPPCropParams(aipp_param_tensor, 0, 0, 0, 0, 0, 0));
            /*
            设置 AIPP（图像预处理引擎）中的填充参数
            int8_t paddingSwitch: 填充开关。非零值-启用填充；0-禁用填充。
            int32_t paddingSizeTop: 顶部填充的大小。
            int32_t paddingSizeBottom: 底部填充的大小。
            int32_t paddingSizeLeft: 左侧填充的大小。
            int32_t paddingSizeRight: 右侧填充的大小。
            uint64_t batchIndex: 批处理参数的索引，通常用于处理批量图像。
            */
            CHECK_ACL(aclmdlSetAIPPPaddingParams(aipp_param_tensor, 0, 0, 0, 0, 0, 0));
            /*
            图像预处理归一化操作过程如下：
            pixel_out_chx(i)=[pixel_in_chx(i)-mean_chn_i-min_chn_i]*var_reci_chn_i
            */
            /*
            sam3预处理要求x' = (x / 255.0f - 0.5f) / 0.5f -->x' = (x  - 0.5f * 255) / 0.5f * 255
            */
            for(int idx = 0; idx < batch_size; idx++){
                float dtcPixelMeanChni0 = SAM3_IMG_MEAN * 255.0;
                float dtcPixelMeanChni1 = SAM3_IMG_MEAN * 255.0;
                float dtcPixelMeanChni2 = SAM3_IMG_MEAN * 255.0;
                CHECK_ACL(aclmdlSetAIPPDtcPixelMean(aipp_param_tensor, dtcPixelMeanChni0, dtcPixelMeanChni1, dtcPixelMeanChni2, 0, idx));
            }
            for(int idx = 0; idx < batch_size; idx++){
                CHECK_ACL(aclmdlSetAIPPDtcPixelMin(aipp_param_tensor, 0.0, 0.0, 0.0, 0.0, idx));
            }
            for(int idx = 0; idx < batch_size; idx++){
                float dtcPixelVarReciChn0 = 1.0 / (SAM3_IMG_STD * 255.0);
                float dtcPixelVarReciChn1 = 1.0 / (SAM3_IMG_STD * 255.0);
                float dtcPixelVarReciChn2 = 1.0 / (SAM3_IMG_STD * 255.0);
                CHECK_ACL(aclmdlSetAIPPPixelVarReci(aipp_param_tensor, dtcPixelVarReciChn0, dtcPixelVarReciChn1, dtcPixelVarReciChn2, 1.0, idx));
            }
            CHECK_ACL(aclmdlSetInputAIPP(model_id, input, aipp_index, aipp_param_tensor));
            CHECK_ACL(aclmdlDestroyAIPP(aipp_param_tensor));
        }
#endif
        std::cout << "intput name: " << name << " type: " << GetType(aclmdlGetInputDataType(model_desc, i)) << " buf_size: " << buf_size;
        aclmdlIODims dims;
        CHECK_ACL(aclmdlGetInputDims(model_desc, i, &dims));
        std::cout << " dims: ";
        for (size_t j = 0; j < dims.dimCount; ++j) {
            std::cout << dims.dims[j];
            if (j + 1 < dims.dimCount) {
                std::cout << " x ";
            }
        }
        std::cout << '\n';
    }

    output_num = aclmdlGetNumOutputs(model_desc);
    for (size_t i = 0; i < output_num; ++i) {
        std::string name = aclmdlGetOutputNameByIndex(model_desc, i);
        size_t buf_size = aclmdlGetOutputSizeByIndex(model_desc, i);
        if (name.find("pred_boxes") != std::string::npos){
            buf_size = batch_size * num_queries * 4 * sizeof(float);
            pred_boxes_idx = i;
        }
        else if(name.find("pred_logits") != std::string::npos){
            buf_size = batch_size * num_queries * sizeof(float);
            pred_logits_idx = i;
        }
        else if(name.find("presence_logits") != std::string::npos){
            buf_size = batch_size * sizeof(float);
            presence_logits_idx = i;
        }
        std::cout << "output name: " << name << " type: " << GetType(aclmdlGetOutputDataType(model_desc, i)) << " buf_size: " << buf_size;
        aclmdlIODims dims;
        CHECK_ACL(aclmdlGetOutputDims(model_desc, i, &dims));
        std::cout << " dims: ";
        for (size_t j = 0; j < dims.dimCount; ++j) {
            std::cout << dims.dims[j];
            if (j + 1 < dims.dimCount) {
                std::cout << " x ";
            }
        }
        std::cout << '\n';

        void *buffer = nullptr;
        void *cpu_buf = nullptr;
        cpu_buf = malloc(buf_size);
        memset(cpu_buf, 0, buf_size);

        CHECK_ACL(aclrtMalloc(&buffer, buf_size, ACL_MEM_MALLOC_NORMAL_ONLY));
        aclDataBuffer* data_buf = aclCreateDataBuffer(buffer, buf_size);
        if (data_buf == nullptr) {
            std::cout << "aclCreateDataBuffer error" << std::endl;
            exit(0);
        }
        CHECK_ACL(aclmdlAddDatasetBuffer(output, data_buf));
        output_buf.push_back(buffer);
        output_size.push_back(buf_size);
        output_data_buf.push_back(data_buf);
        output_cpu.push_back(cpu_buf);
    }
    
    hi_vpc_chn_attr st_chn_attr {};
    st_chn_attr.attr = 0;
    CHECK_DVPP_MPI(hi_mpi_vpc_sys_create_chn(&channel_id_resize, &st_chn_attr));

    // tokenized version of 'person'
    std::vector<int64_t> input_ids={49406,  2533, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407, 49407,
         49407, 49407};
    
    std::vector<int64_t> attention_mask={1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0};
    CHECK_ACL(aclrtMemcpyAsync(input_addr_input_ids, batch_size * 32 * sizeof(int64_t), input_ids.data(), input_ids.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE, stream));
    CHECK_ACL(aclrtMemcpyAsync(input_addr_attention_mask, batch_size * 32 * sizeof(int64_t), attention_mask.data(), attention_mask.size() * sizeof(int64_t), ACL_MEMCPY_HOST_TO_DEVICE, stream));
    CHECK_ACL(aclrtSynchronizeStream(stream));
    auto start = std::chrono::system_clock::now();
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<float> diff;
    float millis_elapsed = 0.0;
    size_t img_size;
    for (const auto& fname : std::filesystem::directory_iterator(in_dir)){
        if(!std::filesystem::is_regular_file(fname.path())){
            continue;
        }
        std::filesystem::path outfile = std::filesystem::path("results") / fname.path().filename();
        cv::Mat img = cv::imread(fname.path());
        if(img_width != img.cols || img_height != img.rows){
            if (input_device){
                CHECK_ACL(aclrtFree(input_device));
                input_device = nullptr;
            }
            img_width  = img.cols;
            img_height = img.rows;
            img_size = img_width * img_height * 3;
            if(input_device == nullptr){
                CHECK_DVPP_MPI(hi_mpi_dvpp_malloc(0, &input_device, img_size));
            }
        }
        CHECK_ACL(aclrtMemcpyAsync(input_device, img_size, img.data, img_size, ACL_MEMCPY_HOST_TO_DEVICE, stream));

        start = std::chrono::system_clock::now();
        data_process_sam3(static_cast<uint8_t*>(input_device), static_cast<uint8_t*>(input_addr_pixel_values), img_width, img_height, 3, input_w, input_h, channel_id_resize, stream);
        CHECK_ACL(aclUpdateDataBuffer(data_buf_inputs[pixel_values_idx], input_addr_pixel_values, batch_size * input_w * input_h * 3));
        CHECK_ACL(aclUpdateDataBuffer(data_buf_inputs[input_ids_idx], input_addr_input_ids, batch_size * 32 * sizeof(int64_t)));
        CHECK_ACL(aclUpdateDataBuffer(data_buf_inputs[attention_mask_idx], input_addr_attention_mask, batch_size * 32 * sizeof(int64_t)));

        CHECK_ACL(aclmdlExecuteAsync(model_id, input, output, stream));

        CHECK_ACL(aclrtMemcpyAsync(output_cpu[pred_boxes_idx], output_size[pred_boxes_idx], output_buf[pred_boxes_idx], output_size[pred_boxes_idx], ACL_MEMCPY_DEVICE_TO_HOST, stream));
        CHECK_ACL(aclrtMemcpyAsync(output_cpu[pred_logits_idx], output_size[pred_logits_idx], output_buf[pred_logits_idx], output_size[pred_logits_idx], ACL_MEMCPY_DEVICE_TO_HOST, stream));
        CHECK_ACL(aclrtMemcpyAsync(output_cpu[presence_logits_idx], output_size[presence_logits_idx], output_buf[presence_logits_idx], output_size[presence_logits_idx], ACL_MEMCPY_DEVICE_TO_HOST, stream));
        
        CHECK_ACL(aclrtSynchronizeStream(stream));
        
        auto detections = postprocess_sam3(output_cpu[pred_boxes_idx], output_cpu[pred_logits_idx], output_cpu[presence_logits_idx], num_queries, img_width, img_height, 0.5f);
        end = std::chrono::system_clock::now();
        diff = end - start;
        millis_elapsed = (diff.count() * 1000);
        std::cout << "millis_elapsed:" << millis_elapsed << std::endl;

        visualize_sam3_results(img, detections, outfile.string());
    }
    CHECK_ACL(aclmdlUnload(model_id));
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(input); ++i) {
        aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(input, i);
        CHECK_ACL(aclDestroyDataBuffer(data_buffer));
        data_buffer = nullptr;
    }
    CHECK_ACL(hi_mpi_dvpp_free(input_addr_pixel_values));
    CHECK_ACL(hi_mpi_dvpp_free(input_addr_input_ids));
    CHECK_ACL(hi_mpi_dvpp_free(input_addr_attention_mask));
#ifdef AIPP_DYNAMIC
    CHECK_ACL(aclrtFree(input_AIPP));
#endif
    CHECK_ACL(hi_mpi_dvpp_free(input_device));
    for (size_t i = 0; i < aclmdlGetDatasetNumBuffers(output); ++i) {
        aclDataBuffer* data_buffer = aclmdlGetDatasetBuffer(output, i);
        void* data = aclGetDataBufferAddr(data_buffer);
        CHECK_ACL(aclrtFree(data));
        CHECK_ACL(aclDestroyDataBuffer(data_buffer));
        data_buffer = nullptr;
    }
    CHECK_ACL(aclmdlDestroyDataset(input));
    CHECK_ACL(aclmdlDestroyDataset(output));
    CHECK_ACL(aclmdlDestroyDesc(model_desc));
    CHECK_DVPP_MPI(hi_mpi_vpc_destroy_chn(channel_id_resize));
    CHECK_ACL(aclrtDestroyStream(stream));
    for (auto& ptr : output_cpu){
        if (ptr){
            free(ptr);
        }
    }
    CHECK_DVPP_MPI(hi_mpi_sys_exit());
    CHECK_ACL(aclFinalize());
    std::cout << "sam3 infer finish" << std::endl;
}