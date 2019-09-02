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

    Halide::Buffer<int32_t> output = gradient.realize(800, 600);

    for (int j = 0; j < output.height(); j++) {
        for (int i = 0; i < output.width(); i++) {
            if (output(i, j) != i+j) {
                printf("Something went wrong!\n"
                    "Pixel (%d,%d) was supposed to be %d, but it's %d\n",
                    i, j, i+j, output(i, j));
                return -1;
            }
        }
    }
    printf("Success!\n");
    return 0;
}
