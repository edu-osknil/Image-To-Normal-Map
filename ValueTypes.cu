#pragma once

class Color{
public:
    float b;
    float g;
    float r;
    __device__ __host__
    Color() : b(0), g(0), r(0) {}
    __device__ __host__
    Color(unsigned int blue, unsigned int green, unsigned int red) : 
        b(blue), g(green), r(red){}
    __device__
    void AddColor(Color newColor){
        b += newColor.b;
        g += newColor.g;
        r += newColor.r;
    }
    __device__
    void Multiply(float amount){
        b *= amount;
        g *= amount;
        r *= amount;
    }
    __device__
    void ApplyToPixel(unsigned char* pixel){
        pixel[0] = (unsigned char)b;
        pixel[1] = (unsigned char)g;
        pixel[2] = (unsigned char)r;
    }
};

struct Vector3{
private:
    __device__
    inline float Distance(){
        return sqrt(__powf(sqrt(x*x+y*y), 2) + z*z);
    }
public:
    float x,y,z;

    __device__
    Vector3(float xVal, float yVal, float zVal) : x(xVal), y(yVal), z(zVal) {}
    __device__ void Normalize(){
        float len = Distance();
        x /= len;
        y /= len;
        z /= len;
    }
};
