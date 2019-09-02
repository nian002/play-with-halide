cmake_minimum_required(VERSION 3.0)

project(halide)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


option(BUILD_WITH_TUTORIALS "Build with tutorials" OFF)
option(BUILD_WITH_EXAMPLES "Build with examples" ON)



# Require C++11 for everything.
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_MACOSX_RPATH ON)


set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")


add_subdirectory(utils)

if (BUILD_WITH_TUTORIALS)
    add_subdirectory(tutorials)
endif()

if (BUILD_WITH_EXAMPLES)
    add_subdirectory(examples)
endif()
