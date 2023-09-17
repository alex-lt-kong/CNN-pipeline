add_subdirectory(swagger/)

add_executable(predict predict.cpp)

target_link_libraries(predict crud)