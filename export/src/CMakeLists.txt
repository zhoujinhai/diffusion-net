
cmake_minimum_required(VERSION 3.9)
project(diffusionnet_infer VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# onnxruntime from thirdPart (1.20.0)
set(ORT_INCLUDE "/home/heygears/zjh/thirdPart/onnxruntime_1.20.0/linux/include")
set(ORT_LIB "/home/heygears/zjh/thirdPart/onnxruntime_1.20.0/linux/lib/libonnxruntime.so")
include_directories(${ORT_INCLUDE})

add_executable(${PROJECT_NAME} main.cpp)

target_compile_options(${PROJECT_NAME} PRIVATE -O3 -fopenmp)
target_link_libraries(${PROJECT_NAME}
    ${ORT_LIB}
    pthread
    gomp
)

add_custom_command(TARGET ${PROJECT_NAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${ORT_LIB}
        $<TARGET_FILE_DIR:${PROJECT_NAME}>
    COMMENT "Copying libonnxruntime.so next to binary"
)
