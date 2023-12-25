add_subdirectory(http_service/)

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(CUDA REQUIRED)
find_package(ImageMagick COMPONENTS Magick++ MagickCore)

include_directories(http_service/)
include_directories(/usr/local/include/torch/)
include_directories(/usr/local/cuda/include/)
include_directories(${ImageMagick_INCLUDE_DIRS})

link_directories(/usr/local/lib/torch/)

include(${PROJECT_SOURCE_DIR}/cmake/proto.cmake)


add_library(model_utils model_utils.cpp)
add_library(global_vars global_vars.cpp ${PROTO_SRCS} ${PROTO_HDRS})

target_link_libraries(model_utils
  opencv_highgui  opencv_core opencv_imgproc opencv_imgcodecs
  torch_cpu c10 c10_cuda)

add_executable(inference_service
  inference_service.cpp
  global_vars.cpp
  utils.cpp
  event_loops.cpp
  ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(inference_service
  http_service model_utils global_vars
  spdlog zmq
  ${ImageMagick_LIBRARIES}
  opencv_highgui  opencv_core opencv_imgproc opencv_imgcodecs
  ${Protobuf_LIBRARIES}
  torch_cpu c10)


add_executable(debug_tools debug_tools.cpp)
target_link_libraries(
  debug_tools model_utils
  opencv_highgui  opencv_core opencv_imgproc opencv_imgcodecs
  torch_cpu c10
  spdlog
)