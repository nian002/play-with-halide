/**
 * Copyright 2019 Xiaobin Wei <xiaobin.wee@gmail.com>
 */

#include <stdio.h>
#include <algorithm>

#include "Halide.h"
#include "halide_image_io.h"


using namespace Halide;
using namespace Halide::Tools;


int main(int argc, char** argv) {
    Var x("x"), y("y"), c("c");

    // Now we'll express a multi-stage pipeline that blurs an image
    // first horizontally, and then vertically.
    {
        Buffer<uint8_t> input = load_image("images/rgb.png");
        Func input_16("input_16");
        input_16(x, y, c) = cast<uint16_t>(input(x, y, c));

        Func blur_x("blur_x");
        blur_x(x, y, c) = (input_16(x-1, y, c) + 2 * input_16(x, y, c) + input_16(x+1, y, c)) / 4;
        Func blur_y("blur_y");
        blur_y(x, y, c) = (blur_x(x, y-1, c) + 2 * blur_x(x, y, c) + blur_x(x, y+1, c)) / 4;

        Func output("output");
        output(x, y, c) = cast<uint8_t>(blur_y(x, y, c));

        // Buffer<uint8_t> result = output.realize(input.width(), input.height(), input.channels());

        Buffer<uint8_t> result(input.width()-2, input.height()-2, input.channels());
        result.set_min(1, 1);
        output.realize(result);

        // Save the result. It should look like a slightly blurry
        // parrot, and it should be two pixels narrower and two pixels
        // shorter than the input image.
        save_image(result, "blurry_parrot_1.png");

        printf("=======================================\n\n\n");
    }

    // The same pipeline, with a boundary condition on the input.
    {
        Buffer<uint8_t> input = load_image("images/rgb.png");

        Func clamped("clamped");
        Expr clamped_x = clamp(x, 0, input.width() - 1);
        Expr clamped_y = clamp(y, 0, input.height() - 1);
        clamped(x, y, c) = input(clamped_x, clamped_y, c);

        // clamped = BoundaryConditions::repeat_edge(input);

        Func input_16("input_16");
        input_16(x, y, c) = cast<uint16_t>(clamped(x, y, c));

        Func blur_x("blur_x");
        blur_x(x, y, c) = (input_16(x-1, y, c) + 2 * input_16(x, y, c) + input_16(x+1, y, c)) / 4;
        Func blur_y("blur_y");
        blur_y(x, y, c) = (blur_x(x, y-1, c) + 2 * blur_x(x, y, c) + blur_x(x, y+1, c)) / 4;

        Func output("output");
        output(x, y, c) = cast<uint8_t>(blur_y(x, y, c));

        Buffer<uint8_t> result = output.realize(input.width(), input.height(), input.channels());

        output.realize(result);

        // Save the result. It should look like a slightly blurry
        // parrot, and it should be two pixels narrower and two pixels
        // shorter than the input image.
        save_image(result, "blurry_parrot_2.png");

        printf("=======================================\n\n\n");
    }


    printf("Success!\n");
    return 0;
}
