#pragma once

__global__
void CudaRemap(
    unsigned char* inImage, 
    int width, 
    int height, 
    int step, 
    int channels
);