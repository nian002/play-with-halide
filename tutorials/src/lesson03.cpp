/**
 * Copyright 2019 Xiaobin Wei <xiaobin.wee@gmail.com>
 */

#include <stdio.h>

#include "Halide.h"


int main(int argc, char** argv) {
    Halide::Func gradient;
    Halide::Var x("x"), y("y");

    Halide::Expr e = x + y;
    gradient(x, y) = e;

    Halide::Buffer<int32_t> output = gradient.realize(8, 8);

    gradient.compile_to_lowered_stmt("gradient.html", {}, Halide::HTML);

    printf("Success!\n");
    return 0;
}
