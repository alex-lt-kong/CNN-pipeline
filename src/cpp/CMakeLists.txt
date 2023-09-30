add_subdirectory(rest/)

find_package(OpenCV REQUIRED)

include_directories(rest/)
include_directories(/usr/local/include/opencv4/)
include_directories(/usr/local/include/torch/)


link_directories(/usr/local/lib/torch/)

add_executable(predict predict.cpp)
target_link_libraries(predict rest spdlog)


add_executable(debug debug.cpp)
target_link_libraries(
  debug opencv_core opencv_imgproc opencv_imgcodecs
  opencv_highgui torch_cpu c10 spdlog
)