/**
 * Copyright 2019 Xiaobin Wei <xiaobin.wee@gmail.com>
 */

#include <stdio.h>
#include <algorithm>

#include "utils.h"

#include "Halide.h"


using namespace Halide;

int main(int argc, char** argv) {
    Var x("x"), y("y");

    // default ordering
    {
        Func gradient("gradient");
        gradient(x, y) = x + y;
        gradient.trace_stores();

        printf("Evaluating gradient row-major\n");
        Buffer<int> output = gradient.realize(4, 4);

        printf("\nEquivalent C:\n");
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                printf("Evaluating at (%d, %d) = %d\n", x, y, x+y);
            }
        }

        printf("\n\nPseudo-code for the schedule:\n");
        gradient.print_loop_nest();
        printf("=======================================\n\n\n");
    }

    // Reorder variables
    {
        Func gradient("gradient_col_major");
        gradient(x, y) = x + y;
        gradient.trace_stores();
        gradient.reorder(y, x);

        printf("Evaluating gradient column-major\n");
        Buffer<int> output = gradient.realize(4, 4);

        printf("\nEquivalent C:\n");
        for (int x = 0; x < 4; x++) {
            for (int y = 0; y < 4; y++) {
                printf("Evaluating at (%d, %d) = %d\n", x, y, x+y);
            }
        }

        printf("\n\nPseudo-code for the schedule:\n");
        gradient.print_loop_nest();
        printf("=======================================\n\n\n");
    }
    // Split a variable into two.
    {
        Func gradient("gradient_split");
        gradient(x, y) = x + y;
        gradient.trace_stores();

        Var x_outer, x_inner;
        gradient.split(x, x_outer, x_inner, 2);

        printf("Evaluating gradient with x split into x_outer and x_inner \n");
        Buffer<int> output = gradient.realize(4, 4);

        printf("\nEquivalent C:\n");
        for (int y = 0; y < 4; y++) {
            for (int x_outer = 0; x_outer < 2; x_outer++) {
                for (int x_inner = 0; x_inner < 2; x_inner++) {
                    int x = x_outer * 2 + x_inner;
                    printf("Evaluating at (%d, %d) = %d\n", x, y, x+y);
                }
            }
        }

        printf("\n\nPseudo-code for the schedule:\n");
        gradient.print_loop_nest();
        printf("=======================================\n\n\n");
    }

    // Fuse two variables into one.
    {
        Func gradient("gradient_fused");
        gradient(x, y) = x + y;
        gradient.trace_stores();

        Var fused;
        gradient.fuse(x, y, fused);

        printf("Evaluating gradient with x and y fused\n");
        Buffer<int> output = gradient.realize(4, 4);

        printf("\nEquivalent C:\n");
        for (int fused = 0; fused < 4*4; fused++) {
            int y = fused / 4;
            int x = fused % 4;
            printf("Evaluating at (%d, %d) = %d\n", x, y, x+y);
        }

        printf("\n\nPseudo-code for the schedule:\n");
        gradient.print_loop_nest();
        printf("=======================================\n\n\n");
    }

    // Evaluating in tiles.
    {
        Func gradient("gradient_tiled");
        gradient(x, y) = x + y;
        gradient.trace_stores();

        Var x_outer, x_inner, y_outer, y_inner;
        // gradient.split(x, x_outer, x_inner, 4);
        // gradient.split(y, y_outer, y_inner, 4);
        // gradient.reorder(x_inner, y_inner, x_outer, y_outer);
        // This pattern is common enough that there's a shorthand for it:
        gradient.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4);

        printf("Evaluating gradient in 4x4 tiles\n");
        Buffer<int> output = gradient.realize(8, 8);

        printf("\nEquivalent C:\n");
        for (int y_outer = 0; y_outer < 2; y_outer++) {
            for (int x_outer = 0; x_outer < 2; x_outer++) {
                for (int y_inner = 0; y_inner < 4; y_inner++) {
                    for (int x_inner = 0; x_inner < 4; x_inner++) {
                        int x = x_outer * 4 + x_inner;
                        int y = y_outer * 4 + y_inner;
                        printf("Evaluating at (%d, %d) = %d\n", x, y, x+y);
                    }
                }
            }
        }

        printf("\n\nPseudo-code for the schedule:\n");
        gradient.print_loop_nest();
        printf("=======================================\n\n\n");
    }

    // Evaluating in vectors.
    {
        Func gradient("gradient_vector");
        gradient(x, y) = x + y;
        gradient.trace_stores();

        // Var x_outer, x_inner;
        // gradient.split(x, x_outer, x_inner, 4);
        // gradient.vectorize(x_inner);
        gradient.vectorize(x, 4);

        printf("Evaluating gradient with x_inner vectorized \n");
        Buffer<int> output = gradient.realize(8, 4);

        printf("\nEquivalent C:\n");
        for (int y = 0; y < 4; y++) {
            for (int x_outer = 0; x_outer < 2; x_outer++) {
                // The loop over x_inner has gone away, and has been
                // replaced by a vectorized version of the
                // expression. On x86 processors, Halide generates SSE
                // for all of this.
                int x_vec[] = {x_outer * 4 + 0,
                               x_outer * 4 + 1,
                               x_outer * 4 + 2,
                               x_outer * 4 + 3};
                int val[] = {x_vec[0] + y,
                             x_vec[1] + y,
                             x_vec[2] + y,
                             x_vec[3] + y};
                printf("Evaluating at <%d, %d, %d, %d>, <%d, %d, %d, %d>:"
                       " <%d, %d, %d, %d>\n",
                       x_vec[0], x_vec[1], x_vec[2], x_vec[3],
                       y, y, y, y,
                       val[0], val[1], val[2], val[3]);
            }
        }

        printf("\n\nPseudo-code for the schedule:\n");
        gradient.print_loop_nest();
        printf("=======================================\n\n\n");
    }

    printf("Success!\n");
    return 0;
}
