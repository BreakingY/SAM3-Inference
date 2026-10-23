#pragma once
#include <iostream>
#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <vector>
#define CUDA_CHECK(callstr)\
    {\
        cudaError_t error_code = callstr;\
        if (error_code != cudaSuccess) {\
            std::cerr << "CUDA error " << error_code << " at " << __FILE__ << ":" << __LINE__;\
            assert(0);\
        }\
    }
// src:BRG(opencv)
void data_process_sam3(uint8_t* src, float* dst, int src_width, int src_height, int src_channels, int dst_width, int dst_height, cudaStream_t &stream);

