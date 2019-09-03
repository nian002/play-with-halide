cmake_minimum_required(VERSION 3.0)

project(halide)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)


option(BUILD_WITH_TUTORIALS "Build with tutorials" ON)
option(BUILD_WITH_EXAMPLES "Build with examples" ON)


set(CMAKE_MACOSX_RPATH ON)

execute_process(COMMAND ${LLVM_CONFIG} --cxxflags OUTPUT_VARIABLE LLVM_CONFIG_CXXFLAGS)
string(FIND "${LLVM_CONFIG_CXXFLAGS}" "-std=c++17" LLVM_CPP17)
string(FIND "${LLVM_CONFIG_CXXFLAGS}" "-std=c++14" LLVM_CPP14)
if (LLVM_CPP17 GREATER -1)
set(CMAKE_CXX_STANDARD 17)
elseif (LLVM_CPP14 GREATER -1)
set(CMAKE_CXX_STANDARD 14)
else()
# Require (at least) C++11 for everything.
set(CMAKE_CXX_STANDARD 11)
endif()

set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)


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


include(CMakePrintHelpers)
cmake_print_variables(LLVM_CPP17)
cmake_print_variables(LLVM_CPP14)