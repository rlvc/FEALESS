#  minimum required cmake version: 3.1.0
cmake_minimum_required(VERSION 3.1.0)

project(FEALESS)

# Save the command line compile commands in the build output
# set(CMAKE_EXPORT_COMPILE_COMMANDS 1)
set(OpenCV_DIR /home/rlvc/Workspace/1_libs/opencv3.4/build)
find_package(realsense2 REQUIRED)
find_package(OpenCV REQUIRED)
include(CheckCXXCompilerFlag)
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
if(COMPILER_SUPPORTS_CXX11)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -std=c11")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
elseif(COMPILER_SUPPORTS_CXX0X)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
endif()

add_executable(FEALESS rs-imshow.cpp)
target_link_libraries(FEALESS ${OpenCV_LIBS} realsense2)
