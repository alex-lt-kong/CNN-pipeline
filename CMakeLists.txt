cmake_minimum_required(VERSION 3.12)

project(vgg16-based-pipeline)

#set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


set(CXXFLAGS "-O2 -Wall -pedantic -Wextra -Wno-unused-parameter")

set(CMAKE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src/cpp/)