/**
 * Copyright 2019 Xiaobin Wei <xiaobin.wee@gmail.com>
 */

#include <stdio.h>
#include <algorithm>

#include "utils.h"

#include "Halide.h"
#include "halide_image_io.h"
#include "utils.h"


using namespace Halide;
using namespace Halide::Tools;


int main(int argc, char** argv) {
    Var x("x"), y("y"), c("c");

    Buffer<uint8_t> input = load_image("images/rgb.png");

    // default ordering
    {
        ScopeTimer t("row-major");
        Func brighten;
        brighten(x, y, c) = cast<uint8_t>(min(input(x, y, c) * 1.5f, 255));
        Buffer<uint8_t> outimg = brighten.realize(input.width(), input.height(),
            input.channels());
        save_image(outimg, "row-major.png");

        printf("=======================================\n\n\n");
    }
    // default ordering
    {
        ScopeTimer t("row-major");
        Func brighten;
        brighten(x, y, c) = cast<uint8_t>(min(input(x, y, c) * 1.5f, 255));
        Buffer<uint8_t> outimg = brighten.realize(input.width(), input.height(),
            input.channels());
        save_image(outimg, "row-major.png");

        printf("=======================================\n\n\n");
    }
    // Reorder variables
    {
        ScopeTimer t("col-major");
        Func brighten;
        brighten(x, y, c) = cast<uint8_t>(min(input(x, y, c) * 1.5f, 255));
        brighten.reorder(y, x, c);
        Buffer<uint8_t> outimg = brighten.realize(input.width(), input.height(),
            input.channels());
        save_image(outimg, "col-major.png");

        printf("=======================================\n\n\n");
    }
    // tiles & vectorize
    {
        ScopeTimer t("tile_vectorize");
        Var x_outer, y_outer, x_inner, y_inner;
        Func brighten;
        brighten(x, y, c) = cast<uint8_t>(min(input(x, y, c) * 1.5f, 255));
        brighten.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4);
        // brighten.vectorize(x_inner);
        Buffer<uint8_t> outimg = brighten.realize(input.width(), input.height(),
            input.channels());
        save_image(outimg, "tiled_vectorized.png");

        printf("=======================================\n\n\n");
    }
    // fusing, tiled and parallel
    {
        ScopeTimer t("tile_fuse_parallel");
        Var x_outer, y_outer, x_inner, y_inner, tile_index;
        Func brighten;
        brighten(x, y, c) = cast<uint8_t>(min(input(x, y, c) * 1.5f, 255));
        brighten.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4);
        brighten.fuse(x_outer, y_outer, tile_index);

        brighten.parallel(tile_index);

        Buffer<uint8_t> outimg = brighten.realize(input.width(), input.height(),
            input.channels());
        save_image(outimg, "tile_fuse_parallel.png");

        printf("=======================================\n\n\n");
    }
    // all togather
    {
        ScopeTimer t("all_togather");
        Var x_outer, y_outer, x_inner, y_inner, tile_index;
        Func brighten;
        brighten(x, y, c) = cast<uint8_t>(min(input(x, y, c) * 1.5f, 255));
        brighten.tile(x, y, x_outer, y_outer, x_inner, y_inner, 64, 64)
                .fuse(x_outer, y_outer, tile_index)
                .parallel(tile_index);

        Var x_inner_outer, y_inner_outer, x_vectors, y_pairs;
        brighten.tile(x_inner, y_inner, x_inner_outer, y_inner_outer, x_vectors, y_pairs, 4, 2)
                .vectorize(x_vectors)
                .unroll(y_pairs);

        Buffer<uint8_t> outimg = brighten.realize(input.width(), input.height(),
            input.channels());
        save_image(outimg, "all_togather.png");

        printf("=======================================\n\n\n");
    }

    printf("Success!\n");
    return 0;
}
