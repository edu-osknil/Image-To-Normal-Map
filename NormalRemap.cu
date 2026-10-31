#include <iostream>
#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>
#include <tuple>
#include "GPUFunc.cuh"

class ImgStorage{
private:
    static std::string GetImgName(std::string path){
        std::string nameStr;
        bool foundType = false;
        for (int i = path.size(); i >= 0; i--){
            if (!foundType){
                if (path[i] == '.')
                    foundType = true;
                continue;
            }
            if (path[i] == '/' || path[i] == '\\')
                break;
            nameStr += path[i];
        }
        std::reverse(nameStr.begin(), nameStr.end());
        return nameStr;
    }
public: 
    static auto ImportImage(){
        while (true)
        {
            std::cout << "Enter a path to the desired image: " << std::endl;
            std::string imagePath;
            std::getline(std::cin, imagePath);
            std::string name = GetImgName(imagePath);

            imagePath.erase(std::remove(imagePath.begin(), imagePath.end(), '"'), imagePath.end());

            cv::Mat image = cv::imread(
                imagePath,
                cv::IMREAD_UNCHANGED
            );
            if (image.empty())
                std::cout << "\033[31mImage loading failed.\033[0m" << std::endl;
            else{
                std::cout << "\033[32mImage loading succesful.\033[0m" << std::endl;
                return std::make_tuple(image, name);
            }
        }
    }
};

class CUDACalls{
public: 
    static void RemapImage(cv::Mat& image){
        unsigned char* gpuImgPtr = nullptr;
        size_t imgSize = image.step * image.rows;
        cudaMalloc(&gpuImgPtr, imgSize);
        cudaMemcpy(gpuImgPtr, image.data, imgSize, cudaMemcpyHostToDevice);
        
        dim3 block(16, 16);
        dim3 grid(
            (image.cols + block.x - 1) / block.x,
            (image.rows + block.y - 1) / block.y
        );

        CudaRemap<<<grid, block>>>(
            gpuImgPtr,
            image.cols,
            image.rows,
            image.step,
            image.channels()
        );
        cudaDeviceSynchronize();

        cudaMemcpy(image.data, gpuImgPtr, imgSize, cudaMemcpyDeviceToHost);
        cudaFree(gpuImgPtr);
    }

    static float* CreateKernel(const int kernelSize, float sigma, int radius){
        float* kernelAdress = nullptr;
        size_t byteSize = kernelSize * kernelSize * sizeof(float);
        cudaMalloc(&kernelAdress, byteSize);
        
        dim3 block(kernelSize, kernelSize);
        dim3 grid(1, 1);
        GenerateGpuKernel<<<grid, block, byteSize>>>(kernelAdress, kernelSize, radius, sigma);
        return kernelAdress;
    }

    static void BlurImage(cv::Mat& image, const unsigned int blurIterations){
        const int kernelSize = 5;
        float sigma = (float)kernelSize / 6;
        int radius = kernelSize / 2;
        float* kernelPointer = CreateKernel(kernelSize, sigma, radius);
        size_t byteData = image.step * image.rows;
        
        unsigned char* deviceInImgAdr = nullptr;
        cudaMalloc(&deviceInImgAdr, byteData);
        cudaMemcpy(deviceInImgAdr, image.data, byteData, cudaMemcpyHostToDevice);

        unsigned char* deviceOutImgAdr = nullptr;
        cudaMalloc(&deviceOutImgAdr, byteData);
        cudaMemcpy(deviceOutImgAdr, image.data, byteData, cudaMemcpyHostToDevice);

        dim3 block(16, 16);
        dim3 grid(
            (image.cols + block.x - 1) / block.x,
            (image.rows + block.y - 1) / block.y
        );
        
        for (unsigned int i = 0; i < blurIterations; i++){
            std::swap(deviceInImgAdr, deviceOutImgAdr);
            CudaBlurImage<<<grid, block>>>(
                deviceInImgAdr,
                deviceOutImgAdr,
                image.cols,
                image.rows,
                image.step,
                radius,
                kernelSize,
                kernelPointer
            );     
        }
        cudaDeviceSynchronize();
        
        cudaMemcpy(image.data, deviceOutImgAdr, byteData, cudaMemcpyDeviceToHost);
        cudaFree(deviceInImgAdr);
        cudaFree(deviceOutImgAdr);
        cudaFree(kernelPointer);
    }
};

int main(){
    std::string folder;
    std::cout<<"Enter destination folder"<<std::endl;
    std::getline(std::cin, folder);
    folder += '\\';
    while (true){
        auto [image, imageName] = ImgStorage::ImportImage();
        CUDACalls::BlurImage(image, 5);
        CUDACalls::RemapImage(image);
        cv::imwrite(
            folder + imageName + ".png",
            image
        );
        while (true){
            std::cout << "Continue? y/n" << std::endl;
            std::string action;
            std::cin >> action;
            if (action == "y")
                break;
            else if (action == "n")
                return 0;
            else    
                std::cout<<"invalid action\n";
        }
        std::cin.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
    }
}