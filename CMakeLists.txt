cmake_minimum_required(VERSION 3.12)

project(VGGNET-based-CNN-pipeline)
add_definitions(-DPROJECT_NAME="${PROJECT_NAME}")

# Always generate compile_commands.json for clangd, etc.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
#set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CXX_STANDARD 20)

include(cmake/compiler-flags.cmake)
include(cmake/sanitizers.cmake)

set(CMAKE_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src/inference/)
