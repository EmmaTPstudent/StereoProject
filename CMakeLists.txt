cmake_minimum_required(VERSION 3.10)
project(StereoProject)

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add the executable
add_executable(program1 main1.cpp)

# Link OpenCV libraries
target_link_libraries(program1 ${OpenCV_LIBS})
