
if (DEFINED ENV{HALIDE_DISTRIB_DIR})
    set(HALIDE_DISTRIB_DIR $ENV{HALIDE_DISTRIB_DIR})
elseif( DEFINED ENV{HALIDE_ROOT_DIR} )
    set(HALIDE_DISTRIB_DIR $ENV{HALIDE_ROOT_DIR})
endif()

if (HALIDE_DISTRIB_DIR)
    include("${HALIDE_DISTRIB_DIR}/halide.cmake")
else()
    message(FATAL_ERROR "HALIDE_DISTRIB_DIR **MUST BE** defined")
endif()


include(CMakePrintHelpers)
cmake_print_variables(HALIDE_DISTRIB_DIR)
cmake_print_variables(HALIDE_ROOT_DIR)
cmake_print_variables(HALIDE_TOOLS_DIR)
cmake_print_variables(HALIDE_INCLUDE_DIR)
cmake_print_variables(HALIDE_COMPILER_LIB)

if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -pthread)
endif()

add_subdirectory(resize)
add_subdirectory(wavelet)



##############################################
# Installation instructions
include(GNUInstallDirs)

install(TARGETS resize
    EXPORT halide_example-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}     # for Windows
)

install(TARGETS wavelet
    EXPORT halide_example-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}     # for Windows
)
