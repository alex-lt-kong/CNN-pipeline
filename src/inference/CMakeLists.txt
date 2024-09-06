# Needed to suppress Magick's nonsense warnings...
add_compile_definitions(MAGICKCORE_QUANTUM_DEPTH=16)
add_compile_definitions(MAGICKCORE_HDRI_ENABLE=0)
add_subdirectory(http_service/)

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(CUDA REQUIRED)
find_package(ImageMagick COMPONENTS Magick++ MagickCore)


include(${PROJECT_SOURCE_DIR}/cmake/proto.cmake)

include_directories(http_service/)
include_directories(/usr/local/include/torch/)
include_directories(/usr/local/cuda/include/)
include_directories(${ImageMagick_INCLUDE_DIRS})

link_directories(/usr/local/lib/torch/)


add_library(model_utils model_utils.cpp)
add_library(utils utils.cpp)
add_library(global_vars global_vars.cpp
${PROTO_SNAP_SRCS} ${PROTO_SNAP_HDRS}
${PROTO_INFER_SRCS} ${PROTO_INFER_HDRS})

target_link_libraries(model_utils
  ${OpenCV_LIBS}
  torch_cpu c10 c10_cuda)

add_executable(inference_service
  inference_service.cpp
  global_vars.cpp
  utils.cpp
  event_loops.cpp
  ${PROTO_SNAP_SRCS} ${PROTO_SNAP_HDRS}
  ${PROTO_INFER_SRCS} ${PROTO_INFER_HDRS})

target_link_libraries(inference_service
  http_service model_utils global_vars
  spdlog zmq
  ${OpenCV_LIBS}
  ${Protobuf_LIBRARIES}
  torch_cpu c10)


add_subdirectory(tools/)