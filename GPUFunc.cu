#pragma once

#include "ValueTypes.cu"

template<typename T>
__device__
inline T Clamp(T t, T a, T b){
    return min(max(a, t), b);
}

__global__
void CudaRemap(unsigned char* image, int width, int height, int step, int channels){
    int idX = blockIdx.x * blockDim.x + threadIdx.x;
    int idY = blockIdx.y * blockDim.y + threadIdx.y;

    if (idX >= width || idY >= height)
        return;
    
    auto clampToEdge = [width](int a) {
        return Clamp(a, 0, width - 1);
    };
    auto clampToTop = [height](int a) {
        return Clamp(a, 0, height - 1);
    };
    auto getPixel = [image, step, channels, clampToEdge](int x, int y) {
        return image + clampToEdge(y) * step + clampToEdge(x) * channels;
    };
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