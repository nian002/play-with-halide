/**
 * Copyright 2019 Xiaobin Wei <xiaobin.wee@gmail.com>
 */

#include <stdio.h>

#include "Halide.h"
#include "halide_image_io.h"


int main(int argc, char** argv) {
    Halide::Buffer<uint8_t> input = Halide::Tools::load_image("images/rgb.png");

    Halide::Var x("x"), y("y"), c("c");
    Halide::Func brighten;
    Halide::Expr value = input(x, y, c);
    value = Halide::cast<float>(value);
    value *= 1.5f;
    value = Halide::min(value, 255.0f);

    value = Halide::cast<uint8_t>(value);
    brighten(x, y, c) = value;

    Halide::Buffer<uint8_t> output = brighten.realize(input.width(),
        input.height(), input.channels());

    Halide::Tools::save_image(output, "brighten.png");

    printf("Success!\n");
    return 0;
}
