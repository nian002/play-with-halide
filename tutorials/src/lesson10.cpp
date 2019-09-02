/**
 * Copyright 2019 Xiaobin Wei <xiaobin.wee@gmail.com>
 */

#include <stdio.h>

#include "HalideBuffer.h"
#include "lesson10_halide.h"

#include "halide_image_io.h"


int main(int argc, char** argv) {

    Halide::Runtime::Buffer<uint8_t> input = Halide::Tools::load_image("images/gray.png");
    const int width = input.width();
    const int height = input.height();
    const int channels = input.channels();

    Halide::Runtime::Buffer<uint8_t> output(width, height);

    int offset = 5;
    int error = brighten(input, offset, output);
    if (error) {
        printf("Halide returned an error: %d\n", error);
        return -1;
    }

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            uint8_t input_val = input(x, y);
            uint8_t output_val = output(x, y);
            uint8_t correct_val = input_val + offset;
            if (output_val != correct_val) {
                printf("output(%d, %d) was %d instead of %d\n",
                       x, y, output_val, correct_val);
                return -1;
            }
        }
    }

    printf("Success!\n");
    return 0;
}
