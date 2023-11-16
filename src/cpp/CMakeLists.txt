add_subdirectory(rest/)

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
find_package(CUDA REQUIRED)
find_package(ImageMagick COMPONENTS Magick++ MagickCore)

include_directories(rest/)
include_directories(/usr/local/include/torch/)
include_directories(/usr/local/cuda/include/)
include_directories(${ImageMagick_INCLUDE_DIRS})

set(MAGICKCORE_QUANTUM_DEPTH 16)

link_directories(/usr/local/lib/torch/)


find_package(Protobuf REQUIRED)
if ( Protobuf_FOUND )
  message( STATUS "Protobuf version: ${Protobuf_VERSION}" )
  message( STATUS "Protobuf include path: ${Protobuf_INCLUDE_DIRS}" )
  message( STATUS "Protobuf libraries: ${Protobuf_LIBRARIES}" )
  message( STATUS "Protobuf lite libraries: ${Protobuf_LITE_LIBRARIES}")
  message( STATUS "Protobuf protoc: ${Protobuf_PROTOC_EXECUTABLE}")
else()
  message( WARNING "Protobuf package not found")
endif()
include_directories(${Protobuf_INCLUDE_DIRS})
include_directories(${CMAKE_CURRENT_BINARY_DIR})
set(PROTO_FILE ${CMAKE_CURRENT_SOURCE_DIR}/proto/snapshot.proto)
protobuf_generate_cpp(PROTO_SRCS PROTO_HDRS ${PROTO_FILE})


add_library(utils utils.cpp)
add_library(model_utils model_utils.cpp)
target_link_libraries(model_utils
  opencv_highgui  opencv_core opencv_imgproc opencv_imgcodecs
  torch_cpu c10 c10_cuda)

add_executable(inference_service
  inference_service.cpp
  global_vars.cpp
  event_loops.cpp
  ${PROTO_SRCS} ${PROTO_HDRS})
target_link_libraries(inference_service
  rest utils model_utils spdlog zmq
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