/**
 * Copyright 2019 Xiaobin Wei <xiaobin.wee@gmail.com>
 */

#include <stdio.h>

#include "Halide.h"


using namespace Halide;


int main(int argc, char** argv) {
    Var x("x"), y("y");
    Func gradient;
    gradient(x, y) = x + y;
    gradient.trace_stores();

    printf("Evaluating gradient from (0, 0) to (7, 7)\n");

    Buffer<int32_t> output(8, 8);
    gradient.realize(output);

    for (int j = 0; j < 8; j++) {
        for (int i = 0; i < 8; i++) {
            if (output(i, j) != i+j) {
                printf("Something went wrong!\n"
                    "Pixel (%d,%d) was supposed to be %d, but it's %d\n",
                    i, j, i+j, output(i, j));
                return -1;
            }
        }
    }


    Buffer<int32_t> shifted(5, 7);
    shifted.set_min(100, 50);

    printf("Evaluating gradient from (100, 50) to (104, 56)\n");
    gradient.realize(shifted);
    for (int y = 50; y < 57; y++) {
        for (int x = 100; x < 105; x++) {
            if (shifted(x, y) != x + y) {
                printf("Something went wrong!\n");
                return -1;
            }
        }
    }

    printf("Success!\n");
    return 0;
}
