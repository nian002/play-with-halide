/**
 * Copyright 2019 Xiaobin Wei <xiaobin.wee@gmail.com>
 */

#include <stdio.h>

#include "Halide.h"

using namespace Halide;


class Brighten : public Halide::Generator<Brighten> {
 public:
    enum class Layout {
        Planar,
        Interleaved,
        Either,
        Specialized,
    };
    GeneratorParam<Layout> layout{
        "layout",
        // default value
        Layout::Planar,
        {
            {
                "planar", Layout::Planar
            },
            {
                "interleaved", Layout::Interleaved
            },
            {
                "either", Layout::Either
            },
            {
                "specialized", Layout::Specialized
            }
        }
    };
    Input<Buffer<uint8_t>> input{"input", 3};
    Output<Buffer<uint8_t>> brighter{"brighter", 3};
    Input<uint8_t> offset{"offset"};
    Var x, y, c;

    void generate() {
        brighter(x, y, c) = input(x, y, c) + offset;
        brighter.vectorize(x, 16);
        if (layout == Layout::Planar) {
            // Do nothing
        } else if (layout == Layout::Interleaved) {
            input.dim(0).set_stride(3)
                .dim(2).set_stride(1);
            brighter.dim(0).set_stride(3)
                .dim(2).set_stride(1);

            input.dim(2).set_bounds(0, 3);
            brighter.dim(2).set_bounds(0, 3);
            brighter.reorder(c, x, y).unroll(c);
        } else if (layout == Layout::Either) {
            input.dim(0).set_stride(Expr());
            brighter.dim(0).set_stride(Expr());
        } else if (layout == Layout::Specialized) {
            input.dim(0).set_stride(Expr());
            brighter.dim(0).set_stride(Expr());

            Expr input_is_planar = (input.dim(0).stride() == 1);
            Expr input_is_interleaved = (
                input.dim(0).stride() ==3 &&
                input.dim(2).stride() == 1
                && input.dim(2).extent() == 3);
            Expr output_is_planar = (brighter.dim(0).stride() == 1);
            Expr output_is_interleaved = (
                brighter.dim(0).stride() ==3 &&
                brighter.dim(2).stride() == 1
                && brighter.dim(2).extent() == 3);

            brighter.specialize(input_is_planar && output_is_planar);
            brighter.specialize(input_is_interleaved && output_is_interleaved)
                .reorder(c, x, y).unroll(c);
        }
    }
};


HALIDE_REGISTER_GENERATOR(Brighten, brighten);
