cmake_minimum_required(VERSION 3.10)
project(MyProject LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV
find_package(OpenCV REQUIRED)

# TensorRT (Hardcoded paths based on user input)
set(TensorRT_ROOT "C:/cudav12.4/TensorRT-10.7.0.23")
set(TensorRT_INCLUDE_DIRS "${TensorRT_ROOT}/include")
set(TensorRT_LIBS "${TensorRT_ROOT}/lib/nvinfer_10.lib")

# Add include directories
include_directories(
    ${OpenCV_INCLUDE_DIRS}
    ${TensorRT_INCLUDE_DIRS}
)

# Link libraries
add_executable(MyExecutable Tracker.cpp)

target_link_libraries(MyExecutable
    ${OpenCV_LIBS}
    ${TensorRT_LIBS}
    cudart
)


# Optional: Set compiler flags (e.g., for C++ standard)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17") # Example: enforce C++17

# (Optional) Link with CUDA's runtime library explicitly if needed
# target_link_libraries(MyExecutable nvinfer) # Might be included in ${TensorRT_LIBS} already

# (Optional) If using specific CUDA libraries:
# target_link_libraries(MyExecutable cudart)
# target_link_libraries(MyExecutable cublas)
# ... etc.


message(STATUS "OpenCV include directories: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "CUDA include directories: ${CUDA_INCLUDE_DIRS}")
message(STATUS "TensorRT include directories: ${TensorRT_INCLUDE_DIRS}")

message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
message(STATUS "CUDA libraries: ${CUDA_LIBRARIES}")
message(STATUS "TensorRT libraries: ${TensorRT_LIBS}")