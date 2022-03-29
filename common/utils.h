#ifndef UTILS_H
#define UTILS_H
#include <opencv2/opencv.hpp>
#include <vector_types.h>
#include <vector_functions.hpp>
#include <sutil/sutil.h>
#include <sutil/CUDAOutputBuffer.h>
void getMatImage(sutil::CUDAOutputBuffer<uchar4> &input, cv::Mat &img);
#endif