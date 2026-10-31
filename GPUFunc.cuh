#pragma once

__global__
void CudaRemap(unsigned char* image, int width, int height, int step, int channels);
__global__
void GenerateGpuKernel(float* globalKernel, int kernelSize, int radius, float sigma);
__global__ 
void CudaBlurImage(
        unsigned char* inImage, 
        unsigned char* outImage, 
        int width, 
        int height, 
        int step,
        int radius,
        int kernelSize,
        float* kernel
    );