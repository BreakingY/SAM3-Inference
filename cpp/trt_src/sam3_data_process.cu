#include "sam3_data_process.cuh"
#define THREAD_COARSENING_FACTOR 2
#define SAM3_RESCALE_FACTOR 0.00392156862745098
#define SAM3_IMG_MEAN 0.5
#define SAM3_IMG_STD 0.5
__global__ void pre_process_sam3(
    uint8_t* src,
    float* dst,
    int src_width,
    int src_height,
    int src_channels,
    int dst_width,
    int dst_height)
{
    int dstX = blockIdx.x*blockDim.x + threadIdx.x;
    int dstY = blockIdx.y*blockDim.y + threadIdx.y;

    int dst_min_x = dstX*THREAD_COARSENING_FACTOR;
    int dst_min_y = dstY*THREAD_COARSENING_FACTOR;

    if (dst_min_x < dst_width && dst_min_y < dst_height)
    {
        int dst_channel_stride = dst_width*dst_height;

        #pragma unroll
        for(int ix=0; ix<THREAD_COARSENING_FACTOR; ix++)
        {
            #pragma unroll
            for (int iy=0; iy < THREAD_COARSENING_FACTOR; iy++)
            {
                int dst_loc_x = dstX*THREAD_COARSENING_FACTOR +ix;
                int dst_loc_y = dstY*THREAD_COARSENING_FACTOR +iy;

                if (dst_loc_x >= dst_width || dst_loc_y >= dst_height)
                {
                    continue;
                    // it is expected that for the vast majority of blocks 
                    // this will not cause thread divergence, except at the edges of the image
                }
                int src_loc_x = src_width*dst_loc_x/dst_width;
                src_loc_x = min(src_loc_x, src_width-1);

                int src_loc_y = src_height*dst_loc_y/dst_height;
                src_loc_y = min(src_loc_y, src_height-1);

                int dst_loc = dst_loc_y*dst_width + dst_loc_x;
                int src_loc = (src_loc_y*src_width + src_loc_x)*src_channels;

                uint8_t sb = src[src_loc];
                uint8_t sg = src[src_loc+1];
                uint8_t sr = src[src_loc+2];

                float dr = (static_cast<float>(sr)*SAM3_RESCALE_FACTOR - SAM3_IMG_MEAN)/SAM3_IMG_STD;
                float dg = (static_cast<float>(sg)*SAM3_RESCALE_FACTOR - SAM3_IMG_MEAN)/SAM3_IMG_STD;
                float db = (static_cast<float>(sb)*SAM3_RESCALE_FACTOR - SAM3_IMG_MEAN)/SAM3_IMG_STD;

                dst[dst_loc] = dr;
                dst[dst_loc + dst_channel_stride] = dg;
                dst[dst_loc + 2*dst_channel_stride] = db;
            }
        }
    }
}
void data_process_sam3(uint8_t* src, float* dst, int src_width, int src_height, int src_channels, int dst_width, int dst_height, cudaStream_t &stream){
    dim3 bsize;
    dim3 gsize;
    bsize.x=16;
    bsize.y=16;
    gsize.x = (dst_width + bsize.x - 1) / (THREAD_COARSENING_FACTOR*bsize.x);
    gsize.y = (dst_height + bsize.y - 1) / (THREAD_COARSENING_FACTOR*bsize.y);
    pre_process_sam3<<<gsize, bsize, 0, stream>>>(src, dst, src_width, src_height, src_channels, dst_width, dst_height);
    cudaError_t run_error = cudaGetLastError();
    if (run_error != cudaSuccess) {
        std::cout << "data_process_sam3 Err" << std::endl;
    }
}