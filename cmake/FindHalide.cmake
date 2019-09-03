
cmake_minimum_required(VERSION 3.0)


if(DEFINED ENV{HALIDE_ROOT_DIR})
    set(HALIDE_ROOT_DIR $ENV{HALIDE_ROOT_DIR})
elseif(DEFINED ENV{HALIDE_DISTRIB_DIR})
    set(HALIDE_ROOT_DIR $ENV{HALIDE_DISTRIB_DIR})
endif()

if(HALIDE_ROOT_DIR)
    find_path(Halide_INCLUDE_DIR
        Halide.h
        HINTS ${HALIDE_ROOT_DIR}/include
    )
    find_path(Halide_TOOLS_DIR
        halide_image.h halide_image_io.h
        HINTS ${HALIDE_ROOT_DIR}/tools
    )

    find_library(Halide_LIBRARY
        NAMES Halide halide
        HINTS ${HALIDE_ROOT_DIR}/lib ${HALIDE_ROOT_DIR}/bin 
    )
else()
    message(FATAL_ERROR "Halide MUST BE installed and HALIDE_ROOT_DIR defined as an Env variable")
endif()

set(Halide_FOUND TRUE)
set(Halide_INCLUDE_DIRS ${Halide_INCLUDE_DIR} ${Halide_TOOLS_DIR})
set(Halide_LIBRARIES ${Halide_LIBRARY})
set(Halide_LIBS ${Halide_LIBRARY})

if(Halide_FOUND AND NOT TARGET Halide::Halide)
    add_library(Halide INTERFACE)
    target_link_libraries(Halide INTERFACE ${Halide_LIBRARY})
    target_include_directories(Halide INTERFACE ${Halide_INCLUDE_DIRS})
    target_compile_features(Halide INTERFACE cxx_std_11)

    add_library(Halide::Halide ALIAS Halide)
endif()

include(CMakePrintHelpers)
cmake_print_variables(HALIDE_ROOT_DIR)
cmake_print_variables(Halide_FOUND)
cmake_print_variables(Halide_INCLUDE_DIR)
cmake_print_variables(Halide_TOOLS_DIR)
cmake_print_variables(Halide_INCLUDE_DIRS)
cmake_print_variables(Halide_LIBRARY)
cmake_print_variables(Halide_LIBRARIES)
cmake_print_variables(Halide_LIBS)
