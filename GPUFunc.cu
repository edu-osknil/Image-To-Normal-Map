#pragma once

#include "ValueTypes.cu"

__global__
void CudaRemap(unsigned char* image, int width, int height, int step, int channels){
    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    int idY = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (idX >= width - 1 || idX <= 1 ||
        idY >= height - 1 || idY <= 1)
        return;
    
    auto getPixel = [image, step, channels](int x, int y) {return image + y * step + x * channels;};
    auto getLuminance = [](unsigned char* pixel) {return 0.299 * pixel[2] + 0.587 * pixel[1] + 0.114 * pixel[0];};

    float dX = getLuminance(getPixel(idX + 1, idY)) - getLuminance(getPixel(idX - 1, idY));
    float dY = getLuminance(getPixel(idX, idY + 1)) - getLuminance(getPixel(idX, idY - 1));

    Vector3 normal = Vector3(dX, dY, 1);
    normal.Normalize();

    unsigned char* pixel = getPixel(idX, idY);
    pixel[0] = 128 * (normal.z + 1);
    pixel[1] = 128 * (normal.y + 1);
    pixel[2] = 128 * (normal.x + 1);
}

__global__
void GenerateGpuKernel(float* globalKernel, int kernelSize, int radius, float sigma){
    extern __shared__ float sharedKernel[];
    const float pi = 3.14159f;
    float sigma2 = sigma * sigma;

    int idX = threadIdx.x - radius;
    int idY = threadIdx.y - radius;
    int kernelIndex = threadIdx.y * kernelSize + threadIdx.x;

    float g = (1.0f / (2 * pi * sigma2)) * __expf(-(idX * idX + idY * idY) / (2 * sigma2));
    sharedKernel[kernelIndex] = g;
    
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0){
        float sum = 0;
        for (int i = 0; i < kernelSize * kernelSize; i++){
            sum += sharedKernel[i];
        }
        for (int i = 0; i < kernelSize * kernelSize; i++){
            sharedKernel[i] /= sum;
        }
    }

    __syncthreads();

    globalKernel[kernelIndex] = sharedKernel[kernelIndex];
}

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
    ){

    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    int idY = blockIdx.y * blockDim.y + threadIdx.y;

    if (idX >= width ||  idY>= height)
        return;

    const int channels = 3;
    Color colorSum = Color();
    for (int y = -radius; y <= radius; y++){
        for (int x = -radius; x <= radius; x++){
            int clampedX = min(max(idX + x, 0), width - 1);
            int clampedY = min(max(idY + y, 0), height - 1);
        
            float kernelValue = kernel[(y + radius) * kernelSize + (x + radius)];
            unsigned char* pixel = inImage + clampedY * step + clampedX * channels;
            Color pixelColor = Color(
                pixel[0],
                pixel[1],
                pixel[2]
            );
            pixelColor.Multiply(kernelValue);
            colorSum.AddColor(pixelColor);
        }
    }
    unsigned char* pixel = outImage + idY * step + idX * channels;
    colorSum.ApplyToPixel(pixel);
}