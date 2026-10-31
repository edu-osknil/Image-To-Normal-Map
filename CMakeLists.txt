cmake_minimum_required(VERSION 3.18)

project(NormalRemap LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

add_executable(NormalRemap NormalRemap.cu ValueTypes.cu GPUFunc.cu)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

target_include_directories(NormalRemap PRIVATE ${OpenCV_INCLUDE_DIRS})
target_link_libraries(NormalRemap PRIVATE ${OpenCV_LIBS})

set_property(TARGET NormalRemap PROPERTY CXX_STANDARD 17)