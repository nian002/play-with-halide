#!/bin/bash

CURRENT_DIR=`pwd`
HOST=`uname -sm`
TARGET_OS=$1
HOST_SYSTEM=`uname`
CPU_CORES=4
if [[ -z ${TARGET_OS} ]]; then
    TARGET_OS=${HOST_SYSTEM}
fi
if [[ ${TARGET_OS} == "Android" || ${TARGET_OS} == "android" ]]; then
    TARGET_OS="Android"
fi

if [[ ${HOST_SYSTEM} == "Linux" ]]; then
    CPU_CORES=`cat /proc/cpuinfo |grep "processor"|wc -l`
elif [[ ${HOST_SYSTEM} == "Darwin" ]]; then
    CPU_CORES=`sysctl -n hw.physicalcpu`
fi

if [[ ! -e "${HALIDE_DISTRIB_DIR}" ]]; then
    export HALIDE_DISTRIB_DIR=${CURRENT_DIR}/out/halide/distrib/${TARGET_OS}
fi

INSTALL_DIR=out/install/${TARGET_OS}
BUILD_ROOT=out/build

echo "*** *** *** *** *** *** *** *** *** *** *** ***"
echo "HALIDE_DISTRIB_DIR: ${HALIDE_DISTRIB_DIR}"
echo "HOST_OS: ${HOST}"
echo "TARGET_OS: ${TARGET_OS}"
echo "CPU_CORES: ${CPU_CORES}"
echo "SRC Dir: ${SRC_DIR}"
echo "Build Dir: ${BUILD_ROOT}"
echo "Install Dir: ${INSTALL_DIR}"
echo "*** *** *** *** *** *** *** *** *** *** *** ***"

if [[ ${TARGET_OS} == "Android" ]]; then
    ANDROID_NDK=`echo $ANDROID_NDK`
    ANDROID_SDK=`echo $ANDROID_SDK`
    if [[ ! -e "${ANDROID_NDK}" ]]; then
        echo "Error: ANDROID_NDK is not set"
        exit
    fi
    if [[ ! -e "${ANDROID_SDK}" ]]; then
        echo "Error: ANDROID_SDK is not set"
        exit
    fi
fi

CMAKE_CONFIG="-G "Ninja" \\
    -D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} \\
    -D CMAKE_POSITION_INDEPENDENT_CODE=ON"
    
if [[ ${TARGET_OS} == "Android" ]]; then
    CMAKE_CONFIG="${CMAKE_CONFIG} \\
    -D CMAKE_TOOLCHAIN_FILE=${ANDROID_NDK}/build/cmake/android.toolchain.cmake \\
    -D ANDROID_SDK=${ANDROID_SDK} \\
    -D ANDROID_NDK=${ANDROID_NDK} \\
    -D ANDROID_PLATFORM=android-28 \\
    -D ANDROID_STL=c++_static \\
    -D ANDROID_PIE=ON "
fi

if [[ ${TARGET_OS} == "Android" ]]; then
    BUILD_DIR=${BUILD_ROOT}/arm
    cmake ${CMAKE_CONFIG} -D ANDROID_ABI="armeabi-v7a with NEON" -B ${BUILD_DIR} .
    cmake --build ${BUILD_DIR} --config Release --target install -- -j${CPU_CORES}

    BUILD_DIR=${BUILD_ROOT}/aarch64
    cmake ${CMAKE_CONFIG} -D ANDROID_ABI="arm64-v8a" -B ${BUILD_DIR} .
    cmake --build ${BUILD_DIR} --config Release --target install -- -j${CPU_CORES}
else
    # Debug vs Release has no difference!!!
    # BUILD_DIR=${BUILD_ROOT}/x64-Debug
    # cmake ${CMAKE_CONFIG} -S ${SRC_DIR} -B ${BUILD_DIR}
    # cmake --build ${BUILD_DIR} --config Debug --target install -- -j${CPU_CORES}

    BUILD_DIR=${BUILD_ROOT}/x64-Release
    cmake ${CMAKE_CONFIG} -D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} -B ${BUILD_DIR} .
    cmake --build ${BUILD_DIR} --config Release --target install -- -j${CPU_CORES}
fi
