cmake_minimum_required(VERSION 3.0)


find_package(Halide REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)


if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -ldl)
endif()

##############################################
# lesson01
add_executable(lesson01 src/lesson01.cpp)
target_link_libraries(lesson01 PRIVATE Halide::Halide)

##############################################
# lesson02
add_executable(lesson02 src/lesson02.cpp)
target_compile_definitions(lesson02 PRIVATE ${PNG_DEFINITIONS})
target_compile_definitions(lesson02 PRIVATE ${JPEG_DEFINITIONS})
target_include_directories(lesson02 PRIVATE ${PNG_INCLUDE_DIRS})
target_include_directories(lesson02 PRIVATE ${JPEG_INCLUDE_DIRS})
target_link_libraries(lesson02 PRIVATE Halide::Halide ${PNG_LIBRARIES} ${JPEG_LIBRARIES})

# cmake_print_variables(PNG_INCLUDE_DIRS)

##############################################
# lesson03
add_executable(lesson03 src/lesson03.cpp)
target_link_libraries(lesson03 PRIVATE Halide::Halide)

##############################################
# lesson04
add_executable(lesson04 src/lesson04.cpp)
target_link_libraries(lesson04 PRIVATE Halide::Halide Halide::utils)

##############################################
# lesson05
add_executable(lesson05 src/lesson05.cpp)
target_link_libraries(lesson05 PRIVATE Halide::Halide Halide::utils)


##############################################
# lesson05-perf
add_executable(lesson05-perf src/lesson05-perf.cpp)
target_compile_definitions(lesson05-perf PRIVATE ${PNG_DEFINITIONS})
target_compile_definitions(lesson05-perf PRIVATE ${JPEG_DEFINITIONS})
target_include_directories(lesson05-perf PRIVATE ${PNG_INCLUDE_DIRS})
target_include_directories(lesson05-perf PRIVATE ${JPEG_INCLUDE_DIRS})
target_link_libraries(lesson05-perf
    PRIVATE
        Halide::Halide
        Halide::utils 
        ${PNG_LIBRARIES}
        ${JPEG_LIBRARIES}
)

##############################################
# lesson06
add_executable(lesson06 src/lesson06.cpp)
target_link_libraries(lesson06 PRIVATE Halide::Halide)


##############################################
# lesson07
add_executable(lesson07 src/lesson07.cpp)
target_compile_definitions(lesson07 PRIVATE ${PNG_DEFINITIONS})
target_compile_definitions(lesson07 PRIVATE ${JPEG_DEFINITIONS})
target_include_directories(lesson07 PRIVATE ${PNG_INCLUDE_DIRS})
target_include_directories(lesson07 PRIVATE ${JPEG_INCLUDE_DIRS})
target_link_libraries(lesson07 PRIVATE Halide::Halide ${PNG_LIBRARIES} ${JPEG_LIBRARIES})

##############################################
# lesson08
add_executable(lesson08 src/lesson08.cpp)
target_link_libraries(lesson08 PRIVATE Halide::Halide)


##############################################
# lesson10
add_executable(lesson10_run src/lesson10_generate.cpp)
target_link_libraries(lesson10_run PRIVATE Halide::Halide)

# add_custom_command(
#     TARGET lesson10_run
#     POST_BUILD
#     COMMAND lesson10_run
#     COMMENT Generate the halide intermediate library
# )
set(GENERATED_FILES lesson10_halide.h lesson10_halide.a)
add_custom_command(
    OUTPUT ${GENERATED_FILES}
    COMMAND lesson10_run
)
add_custom_target(lesson10_generate DEPENDS ${GENERATED_FILES})
# add_library(lesson10_halide INTERFACE IMPORTED)
# set_target_properties(lesson10_halide PROPERTIES INTERFACE_IMPORTED_LOCATION ${CMAKE_CURRENT_BINARY_DIR}/lesson10_halide.a)
add_library(lesson10_halide INTERFACE)
target_include_directories(lesson10_halide INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(lesson10_halide INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/lesson10_halide.a)

add_executable(lesson10 src/lesson10.cpp)
target_compile_definitions(lesson10 PRIVATE ${PNG_DEFINITIONS})
target_compile_definitions(lesson10 PRIVATE ${JPEG_DEFINITIONS})
target_include_directories(lesson10 PRIVATE ${PNG_INCLUDE_DIRS})
target_include_directories(lesson10 PRIVATE ${JPEG_INCLUDE_DIRS})
target_include_directories(lesson10 PRIVATE ${CMAKE_CURRENT_BINARY_DIR})    # generated files
target_link_libraries(lesson10
    PRIVATE
        Halide::Halide 
        ${PNG_LIBRARIES} 
        ${JPEG_LIBRARIES}
        lesson10_halide
)
add_dependencies(lesson10 lesson10_generate)


##############################################
# lesson16
add_executable(lesson16_generate src/lesson16_generate.cpp ../halide/tools/GenGen.cpp)
target_link_libraries(lesson16_generate PRIVATE Halide::Halide)
target_compile_options(lesson16_generate PRIVATE -fno-rtti)

set(PLANAR_GENERATED_FILES brighten_planar.h brighten_planar.a)
add_custom_command(
    OUTPUT ${PLANAR_GENERATED_FILES}
    COMMAND lesson16_generate -g brighten -o ${CMAKE_CURRENT_BINARY_DIR} -f brighten_planar layout=planar target=host
)
add_custom_target(planar_generated DEPENDS ${PLANAR_GENERATED_FILES})
add_library(brighten_planar INTERFACE)
target_include_directories(brighten_planar INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(brighten_planar INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/brighten_planar.a)

set(INTERLEAVED_GENERATED_FILES brighten_interleaved.h brighten_interleaved.a)
add_custom_command(
    OUTPUT ${INTERLEAVED_GENERATED_FILES}
    COMMAND lesson16_generate -g brighten -o ${CMAKE_CURRENT_BINARY_DIR} -f brighten_interleaved layout=interleaved target=host
)
add_custom_target(interleaved_generated DEPENDS ${INTERLEAVED_GENERATED_FILES})
add_library(brighten_interleaved INTERFACE)
target_include_directories(brighten_interleaved INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(brighten_interleaved INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/brighten_interleaved.a)

set(EITHER_GENERATED_FILES brighten_either.h brighten_either.a)
add_custom_command(
    OUTPUT ${EITHER_GENERATED_FILES}
    COMMAND lesson16_generate -g brighten -o ${CMAKE_CURRENT_BINARY_DIR} -f brighten_either layout=either target=host
)
add_custom_target(either_generated DEPENDS ${EITHER_GENERATED_FILES})
add_library(brighten_either INTERFACE)
target_include_directories(brighten_either INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(brighten_either INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/brighten_either.a)

set(SPECIALIZED_GENERATED_FILES brighten_specialized.h brighten_specialized.a)
add_custom_command(
    OUTPUT ${SPECIALIZED_GENERATED_FILES}
    COMMAND lesson16_generate -g brighten -o ${CMAKE_CURRENT_BINARY_DIR} -f brighten_specialized layout=specialized target=host
)
add_custom_target(specialized_generated DEPENDS ${SPECIALIZED_GENERATED_FILES})
add_library(brighten_specialized INTERFACE)
target_include_directories(brighten_specialized INTERFACE ${CMAKE_CURRENT_BINARY_DIR})
target_link_libraries(brighten_specialized INTERFACE ${CMAKE_CURRENT_BINARY_DIR}/brighten_specialized.a)

add_executable(lesson16 src/lesson16_image.cpp)
target_link_libraries(lesson16
    PRIVATE
        Halide::Halide
        Halide::utils
        brighten_planar
        brighten_interleaved
        brighten_either
        brighten_specialized
        ${PNG_LIBRARIES}
        ${JPEG_LIBRARIES}
)
target_compile_definitions(lesson16 PRIVATE ${PNG_DEFINITIONS})
target_compile_definitions(lesson16 PRIVATE ${JPEG_DEFINITIONS})
target_include_directories(lesson16 PRIVATE ${PNG_INCLUDE_DIRS})
target_include_directories(lesson16 PRIVATE ${JPEG_INCLUDE_DIRS})
# make sure lesson16 build after it's dependencies are built
add_dependencies(lesson16 planar_generated interleaved_generated either_generated specialized_generated)

##############################################
# Installation instructions
include(GNUInstallDirs)

install(TARGETS lesson01
    EXPORT halide_tutorials-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}     # for Windows
)

install(TARGETS lesson02
    EXPORT halide_tutorials-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}     # for Windows
)
install(DIRECTORY ${CMAKE_SOURCE_DIR}/images/ DESTINATION ${CMAKE_INSTALL_BINDIR}/images)

install(TARGETS lesson03
    EXPORT halide_tutorials-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}     # for Windows
)

install(TARGETS lesson04
    EXPORT halide_tutorials-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}     # for Windows
)

install(TARGETS lesson05
    EXPORT halide_tutorials-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}     # for Windows
)

install(TARGETS lesson05-perf
    EXPORT halide_tutorials-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}     # for Windows
)

install(TARGETS lesson06
    EXPORT halide_tutorials-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}     # for Windows
)

install(TARGETS lesson07
    EXPORT halide_tutorials-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}     # for Windows
)

install(TARGETS lesson08
    EXPORT halide_tutorials-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}     # for Windows
)

install(TARGETS lesson10
    EXPORT halide_tutorials-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}     # for Windows
)

install(TARGETS lesson16
    EXPORT halide_tutorials-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}     # for Windows
)
