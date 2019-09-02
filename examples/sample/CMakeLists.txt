
project(sample)
cmake_minimum_required(VERSION 3.0)


add_executable(sample sample.cpp)
set_target_properties(sample PROPERTIES CXX_STANDARD 11 CXX_STANDARD_REQUIRED YES CXX_EXTENSIONS NO)
target_include_directories(sample PRIVATE "${HALIDE_INCLUDE_DIR}" "${HALIDE_TOOLS_DIR}")
halide_use_image_io(sample)

# Define a halide_library() for each generator we have, and link each one into sample
file(GLOB GENS RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/*_generator.cpp")
foreach(GEN_SRC ${GENS})
    string(REPLACE "_generator.cpp" "" GEN_NAME "${GEN_SRC}")
    halide_library("${GEN_NAME}" SRCS ${GEN_SRC})
    target_link_libraries(sample PUBLIC "${GEN_NAME}")
endforeach()
