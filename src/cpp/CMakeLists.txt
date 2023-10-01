add_subdirectory(rest/)

find_package(OpenCV REQUIRED)

include_directories(rest/)
#include_directories(/usr/local/include/opencv4/)
include_directories(/usr/local/include/torch/)


link_directories(/usr/local/lib/torch/)

add_library(utils utils.cpp)
add_library(model_utils model_utils.cpp)
target_link_libraries(model_utils
  opencv_highgui  opencv_core opencv_imgproc opencv_imgcodecs
  torch_cpu c10)

add_executable(predict predict.cpp)
target_link_libraries(predict
  rest utils model_utils spdlog zmq
  opencv_highgui  opencv_core opencv_imgproc opencv_imgcodecs
  torch_cpu c10)


add_executable(debug debug.cpp)
target_link_libraries(
  debug model_utils
  opencv_highgui  opencv_core opencv_imgproc opencv_imgcodecs
  torch_cpu c10
  spdlog
)