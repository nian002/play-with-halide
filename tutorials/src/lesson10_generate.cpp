/**
 * Copyright 2019 Xiaobin Wei <xiaobin.wee@gmail.com>
 */

#include <stdio.h>

#include "Halide.h"

using namespace Halide;


int main(int argc, char** argv) {

    Param<uint8_t> offset;
    ImageParam input(type_of<uint8_t>(), 2);

    Var x("x"), y("y");
    Func brighten;

    brighten(x, y) = input(x, y) + offset;
    brighten.vectorize(x, 16).parallel(y);

    brighten.compile_to_static_library("lesson10_halide", {input, offset}, "brighten");

    printf("Halide pipeline compiled, but not yet run.\n");

    return 0;
}
