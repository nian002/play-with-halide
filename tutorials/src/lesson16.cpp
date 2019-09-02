/**
 * Copyright 2019 Xiaobin Wei <xiaobin.wee@gmail.com>
 */

#include <stdio.h>
#include <assert.h>

#include "utils.h"

#include "HalideBuffer.h"

#include "brighten_planar.h"
#include "brighten_interleaved.h"
#include "brighten_either.h"
#include "brighten_specialized.h"


int main(int argc, char** argv) {
    Halide::Runtime::Buffer<uint8_t> planar_input(1024, 768, 3);
    Halide::Runtime::Buffer<uint8_t> planar_output(1024, 768, 3);
    Halide::Runtime::Buffer<uint8_t> interleaved_input =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(1024, 768, 3);
    Halide::Runtime::Buffer<uint8_t> interleaved_output =
        Halide::Runtime::Buffer<uint8_t>::make_interleaved(1024, 768, 3);

    assert(planar_output.dim(0).stride() == 1);
    assert(interleaved_input.dim(0).stride() == 3);
    assert(interleaved_output.dim(0).stride() == 3);
    assert(interleaved_input.dim(2).stride() == 1);
    assert(interleaved_output.dim(2).stride() == 1);

    ScopeTimer timer("planar");
    for (int i = 0; i < 1000; i++) {
        brighten_planar(planar_input, 1, planar_output);
    }

    timer.reset("interleaved");
    for (int i = 0; i < 1000; i++) {
        brighten_interleaved(interleaved_input, 1, interleaved_output);
    }

    timer.reset("either on plannar image");
    for (int i = 0; i < 1000; i++) {
        brighten_either(planar_input, 1, planar_output);
    }

    timer.reset("either on interleaved image");
    for (int i = 0; i < 1000; i++) {
        brighten_either(interleaved_input, 1, interleaved_output);
    }

    timer.reset("specialized on plannar image");
    for (int i = 0; i < 1000; i++) {
        brighten_specialized(planar_input, 1, planar_output);
    }

    timer.reset("specialized on interleaved image");
    for (int i = 0; i < 1000; i++) {
        brighten_specialized(interleaved_input, 1, interleaved_output);
    }

    return 0;
}
