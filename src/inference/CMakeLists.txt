# Needed to suppress Magick's nonsense warnings...
add_compile_definitions(MAGICKCORE_QUANTUM_DEPTH=16)
add_compile_definitions(MAGICKCORE_HDRI_ENABLE=0)

# Torch is NOT managed by vcpkg, we need to manually specify its root dir
set(TORCH_HOME "/apps/libtorch")

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(cppzmq CONFIG REQUIRED)
find_package(CUDA REQUIRED)
set(Torch_DIR "${TORCH_HOME}/share/cmake/Torch")
find_package(Torch REQUIRED) #Dont include this, we directly pass include_directories() and link_directories()
find_package(ImageMagick COMPONENTS Magick++ MagickCore)


include(${PROJECT_SOURCE_DIR}/cmake/proto.cmake)

include_directories("${TORCH_HOME}/include")
include_directories("${TORCH_HOME}/include/torch/csrc/api/include")
include_directories(${ImageMagick_INCLUDE_DIRS})

link_directories("${TORCH_HOME}/lib")


add_library(model_utils model_utils.cpp
        ${PROTO_SNAP_SRCS} ${PROTO_SNAP_HDRS}
        ${PROTO_INFER_SRCS} ${PROTO_INFER_HDRS})
add_library(utils utils.cpp)
add_library(global_vars global_vars.cpp
        ${PROTO_SNAP_SRCS} ${PROTO_SNAP_HDRS}
        ${PROTO_INFER_SRCS} ${PROTO_INFER_HDRS})

target_link_libraries(model_utils
        ${OpenCV_LIBS}
        torch_cuda)

add_executable(inference_service
        inference_service.cpp
        global_vars.cpp
        utils.cpp
        event_loops.cpp
        ${PROTO_SNAP_SRCS} ${PROTO_SNAP_HDRS}
        ${PROTO_INFER_SRCS} ${PROTO_INFER_HDRS})

target_link_libraries(inference_service
        model_utils global_vars
        spdlog::spdlog libzmq
        ${OpenCV_LIBS}
        ${Protobuf_LIBRARIES}
        torch_cuda torch_cpu c10)


add_subdirectory(tools/)