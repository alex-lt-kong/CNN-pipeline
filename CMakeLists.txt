cmake_minimum_required(VERSION 3.12)

project(VGGNET-based-CNN-pipeline)

# Always generate compile_commands.json for clangd, etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 17)


set(CXXFLAGS "-O2 -Wall -pedantic -Wextra -Wno-unused-parameter")

set(CMAKE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src/cpp/)
