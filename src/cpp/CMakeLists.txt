add_subdirectory(rest/)

find_package(OpenCV REQUIRED)
find_package(spdlog REQUIRED)
#find_package(ImageMagick COMPONENTS Magick++)
find_package(ImageMagick COMPONENTS Magick++ MagickCore)

include_directories(rest/)
include_directories(/usr/local/include/torch/)
include_directories(${ImageMagick_INCLUDE_DIRS})

set(MAGICKCORE_QUANTUM_DEPTH 16)

link_directories(/usr/local/lib/torch/)

add_library(utils utils.cpp)
add_library(model_utils model_utils.cpp)
target_link_libraries(model_utils
  opencv_highgui  opencv_core opencv_imgproc opencv_imgcodecs
  torch_cpu c10)

add_executable(predict predict.cpp)
target_link_libraries(predict
  rest utils model_utils spdlog zmq ${ImageMagick_LIBRARIES}
  opencv_highgui  opencv_core opencv_imgproc opencv_imgcodecs
  torch_cpu c10)


add_executable(debug debug.cpp)
target_link_libraries(
  debug model_utils
  opencv_highgui  opencv_core opencv_imgproc opencv_imgcodecs
  torch_cpu c10
  spdlog
)