cmake_minimum_required(VERSION 2.8)
#OPENCV
FIND_PACKAGE(OpenCV HINTS "C:/opencv320/opencv/build")
INCLUDE_DIRECTORIES(${OpenCV_INCLUDE_DIRS} )
link_directories(${OPENCV_LIBRARY_DIRS})
add_definitions(${OPENCV_DEFINITIONS})

add_subdirectory(cudatool)

include_directories(${PROJECT_SOURCE_DIR}/src/cudaTool) 
include_directories(${PROJECT_SOURCE_DIR}/cudaTool) 

add_executable(depthDenoise main.cpp depth_denoising.cpp depth_denoising.h cloudream_pinhole_camera.h)

TARGET_LINK_LIBRARIES(depthDenoise cudatool ${OpenCV_LIBS})