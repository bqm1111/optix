#include "utils.h"
void getMatImage(sutil::CUDAOutputBuffer<uchar4> &input, cv::Mat &img)
{
    sutil::ImageBuffer buffer;
    buffer.data = input.getHostPointer();
    buffer.width = input.width();
    buffer.height = input.height();
    buffer.pixel_format = sutil::BufferImageFormat::UNSIGNED_BYTE4;
    for (int j = buffer.height - 1; j >= 0; --j)
    {
        for (int i = 0; i < buffer.width; ++i)
        {
            const int32_t dst_idx = 3 * buffer.width * (buffer.height - j - 1) + 3 * i;
            const int32_t src_idx = 4 * buffer.width * j + 4 * i;
            img.data[dst_idx + 0] = reinterpret_cast<uint8_t *>(buffer.data)[src_idx + 0];
            img.data[dst_idx + 1] = reinterpret_cast<uint8_t *>(buffer.data)[src_idx + 1];
            img.data[dst_idx + 2] = reinterpret_cast<uint8_t *>(buffer.data)[src_idx + 2];
        }
    }
}
