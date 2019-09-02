cmake_minimum_required(VERSION 3.0)


add_library(utils STATIC src/utils.cpp)
add_library(Halide::utils ALIAS utils)

target_compile_features(utils PRIVATE cxx_std_11)
target_compile_options(utils PRIVATE $<$<CXX_COMPILER_ID:GNU>:-Wall>)

target_include_directories(utils
    PUBLIC
        $<INSTALL_INTERFACE:include>
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)
