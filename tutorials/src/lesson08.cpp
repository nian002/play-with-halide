/**
 * Copyright 2019 Xiaobin Wei <xiaobin.wee@gmail.com>
 */

#include <stdio.h>

#include "Halide.h"

using namespace Halide;


int main(int argc, char** argv) {
    Var x("x"), y("y");

    // Let's examine various scheduling options for a simple two stage
    // pipeline. We'll start with the default schedule:
    {
        Func producer("producer_default"), consumer("consumer_default");
        producer(x, y) = sin(x * y);
        consumer(x, y) = (producer(x, y) + producer(x, y+1) +
            producer(x+1, y) + producer(x+1, y+1)) / 4;
        producer.trace_stores();
        consumer.trace_stores();
        printf("\nEvaluating producer-consumer pipeline with default schedule\n");
        consumer.realize(4, 4);

        // The equivalent C code is:
        float result[4][4];
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                result[y][x] = (sin(x*y) +
                                sin(x*(y+1)) +
                                sin((x+1)*y) +
                                sin((x+1)*(y+1)))/4;
            }
        }
        printf("\nPseudo-code for the schedule:\n");
        consumer.print_loop_nest();
    }

    // Next we'll examine the next simplest option - computing all
    // values required in the producer before computing any of the
    // consumer. We call this schedule "root".
    {
        Func producer("producer_root"), consumer("consumer_root");
        producer(x, y) = sin(x * y);
        consumer(x, y) = (producer(x, y) + producer(x, y+1) +
            producer(x+1, y) + producer(x+1, y+1)) / 4;

        // Tell Halide to evaluate all of producer before any of consumer.
        producer.compute_root();

        producer.trace_stores();
        consumer.trace_stores();

        printf("\n\nEvaluating producer.compute_root()\n");
        consumer.realize(4, 4);

        // The equivalent C code is:
        float result[4][4];
        // Allocate some temporary storage for the producer.
        float producer_storage[5][5];
        // Compute the producer.
        for (int y = 0; y < 5; y++) {
            for (int x = 0; x < 5; x++) {
                producer_storage[y][x] = sin(x * y);
            }
        }
        // Compute the consumer. Skip the prints this time.
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {
                result[y][x] = (producer_storage[y][x] +
                                producer_storage[y+1][x] +
                                producer_storage[y][x+1] +
                                producer_storage[y+1][x+1])/4;
            }
        }
        printf("\nPseudo-code for the schedule:\n");
        consumer.print_loop_nest();
    }

    // We can make choices in between full inlining and
    // compute_root. Next we'll alternate between computing the
    // producer and consumer on a per-scanline basis:
    {
        Func producer("producer_y"), consumer("consumer_y");
        producer(x, y) = sin(x * y);
        consumer(x, y) = (producer(x, y) + producer(x, y+1) +
            producer(x+1, y) + producer(x+1, y+1)) / 4;

        // Tell Halide to evaluate producer as needed per y coordinate
        // of the consumer:
        producer.compute_at(consumer, y);

        producer.trace_stores();
        consumer.trace_stores();

        printf("\n\nEvaluating producer.compute_at(consumer, y)\n");
        consumer.realize(4, 4);

        float result[4][4];

        // There's an outer loop over scanlines of consumer:
        for (int y = 0; y < 4; y++) {

            // Allocate space and compute enough of the producer to
            // satisfy this single scanline of the consumer. This
            // means a 5x2 box of the producer.
            float producer_storage[2][5];
            for (int py = y; py < y + 2; py++) {
                for (int px = 0; px < 5; px++) {
                    producer_storage[py-y][px] = sin(px * py);
                }
            }

            // Compute a scanline of the consumer.
            for (int x = 0; x < 4; x++) {
                result[y][x] = (producer_storage[0][x] +
                                producer_storage[1][x] +
                                producer_storage[0][x+1] +
                                producer_storage[1][x+1])/4;
            }
        }

        printf("\nPseudo-code for the schedule:\n");
        consumer.print_loop_nest();
    }

    // We could also say producer.compute_at(consumer, x), but this
    // would be very similar to full inlining (the default
    // schedule). Instead let's distinguish between the loop level at
    // which we allocate storage for producer, and the loop level at
    // which we actually compute it. This unlocks a few optimizations.
    {
        Func producer("producer_root_y"), consumer("consumer_root_y");
        producer(x, y) = sin(x * y);
        consumer(x, y) = (producer(x, y) + producer(x, y+1) +
            producer(x+1, y) + producer(x+1, y+1)) / 4;

        // Tell Halide to make a buffer to store all of producer at
        // the outermost level:
        producer.store_root();
        // ... but compute it as needed per y coordinate of the
        // consumer.
        producer.compute_at(consumer, y);

        producer.trace_stores();
        consumer.trace_stores();

        printf("\n\nEvaluating producer.store_root().compute_at(consumer, y)\n");
        consumer.realize(4, 4);

        float result[4][4];

        // producer.store_root() implies that storage goes here:
        // float producer_storage[5][5];
        // Actually store 2 scanlines instead of 5
        float producer_storage[2][5];

        // There's an outer loop over scanlines of consumer:
        for (int y = 0; y < 4; y++) {
            // Compute enough of the producer to satisfy this scanline
            // of the consumer.
            for (int py = y; py < y + 2; py++) {

                // Skip over rows of producer that we've already
                // computed in a previous iteration.
                if (y > 0 && py == y) continue;

                for (int px = 0; px < 5; px++) {
                    producer_storage[py & 1][px] = sin(px * py);
                }
            }

            // Compute a scanline of the consumer.
            for (int x = 0; x < 4; x++) {
                result[y][x] = (producer_storage[y & 1][x] +
                                producer_storage[(y+1) & 1][x] +
                                producer_storage[y & 1][x+1] +
                                producer_storage[(y+1) & 1][x+1])/4;
            }

            printf("\nPseudo-code for the schedule:\n");
            consumer.print_loop_nest();
        }
    }

    // We can do even better, by leaving the storage outermost, but
    // moving the computation into the innermost loop:
    {
        Func producer("producer_root_x"), consumer("consumer_root_x");
        producer(x, y) = sin(x * y);
        consumer(x, y) = (producer(x, y) +
                        producer(x, y+1) +
                        producer(x+1, y) +
                        producer(x+1, y+1))/4;

        // Store outermost, compute innermost.
        producer.store_root().compute_at(consumer, x);

        producer.trace_stores();
        consumer.trace_stores();

        printf("\n\nEvaluating producer.store_root().compute_at(consumer, x)\n");
        consumer.realize(4, 4);

        float result[4][4];

        // producer.store_root() implies that storage goes here, but
        // we can fold it down into a circular buffer of two
        // scanlines:
        float producer_storage[2][5];

        // For every pixel of the consumer:
        for (int y = 0; y < 4; y++) {
            for (int x = 0; x < 4; x++) {

                // Compute enough of the producer to satisfy this
                // pixel of the consumer, but skip values that we've
                // already computed:
                if (y == 0 && x == 0)
                    producer_storage[y & 1][x] = sin(x*y);
                if (y == 0)
                    producer_storage[y & 1][x+1] = sin((x+1)*y);
                if (x == 0)
                    producer_storage[(y+1) & 1][x] = sin(x*(y+1));
                producer_storage[(y+1) & 1][x+1] = sin((x+1)*(y+1));

                result[y][x] = (producer_storage[y & 1][x] +
                                producer_storage[(y+1) & 1][x] +
                                producer_storage[y & 1][x+1] +
                                producer_storage[(y+1) & 1][x+1])/4;
            }
        }

        printf("\nPseudo-code for the schedule:\n");
        consumer.print_loop_nest();
    }

    // We're running out of options. We can make new ones by
    // splitting. We can store_at or compute_at at the natural
    // variables of the consumer (x and y), or we can split x or y
    // into new inner and outer sub-variables and then schedule with
    // respect to those. We'll use this to express fusion in tiles:
    {
        Func producer("producer_tile"), consumer("consumer_tile");
        producer(x, y) = sin(x * y);
        consumer(x, y) = (producer(x, y) +
                        producer(x, y+1) +
                        producer(x+1, y) +
                        producer(x+1, y+1))/4;

        // We'll compute 8x8 of the consumer, in 4x4 tiles.
        Var x_outer, y_outer, x_inner, y_inner;
        consumer.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4);

        // Compute the producer per tile of the consumer
        producer.compute_at(consumer, x_outer);

        // Notice that I wrote my schedule starting from the end of
        // the pipeline (the consumer). This is because the schedule
        // for the producer refers to x_outer, which we introduced
        // when we tiled the consumer. You can write it in the other
        // order, but it tends to be harder to read.

        // Turn on tracing.
        producer.trace_stores();
        consumer.trace_stores();
        printf("\n\nEvaluating:\n"
            "consumer.tile(x, y, x_outer, y_outer, x_inner, y_inner, 4, 4);\n"
            "producer.compute_at(consumer, x_outer);\n");
        consumer.realize(8, 8);

        float result[8][8];
        // For every tile of the consumer:
        for (int y_outer = 0; y_outer < 2; y_outer++) {
            for (int x_outer = 0; x_outer < 2; x_outer++) {
                // Compute the x and y coords of the start of this tile.
                int x_base = x_outer*4;
                int y_base = y_outer*4;

                // Compute enough of producer to satisfy this tile. A
                // 4x4 tile of the consumer requires a 5x5 tile of the
                // producer.
                float producer_storage[5][5];
                for (int py = y_base; py < y_base + 5; py++) {
                    for (int px = x_base; px < x_base + 5; px++) {
                        producer_storage[py-y_base][px-x_base] = sin(px * py);
                    }
                }

                // Compute this tile of the consumer
                for (int y_inner = 0; y_inner < 4; y_inner++) {
                    for (int x_inner = 0; x_inner < 4; x_inner++) {
                        int x = x_base + x_inner;
                        int y = y_base + y_inner;
                        result[y][x] =
                            (producer_storage[y - y_base][x - x_base] +
                             producer_storage[y - y_base + 1][x - x_base] +
                             producer_storage[y - y_base][x - x_base + 1] +
                             producer_storage[y - y_base + 1][x - x_base + 1])/4;
                    }
                }
            }
        }
        printf("Pseudo-code for the schedule:\n");
        consumer.print_loop_nest();
    }


    // Let's try a mixed strategy that combines what we have done with
    // splitting, parallelizing, and vectorizing. This is one that
    // often works well in practice for large images. If you
    // understand this schedule, then you understand 95% of scheduling
    // in Halide.
    {
        Func producer("producer_mixed"), consumer("consumer_mixed");
        producer(x, y) = sin(x * y);
        consumer(x, y) = (producer(x, y) +
                        producer(x, y+1) +
                        producer(x+1, y) +
                        producer(x+1, y+1))/4;
        Var yo, yi;
        consumer.split(y, yo, yi, 16).parallel(yo).vectorize(x, 4);

        producer.store_at(consumer, yo).compute_at(consumer, yi).vectorize(x, 4);

        // Let's leave tracing off this time, because we're going to
        // evaluate over a larger image.
        // consumer.trace_stores();
        // producer.trace_stores();

        printf("\n\nEvaluating producer_mixed, consumer_mixed\n");
        Buffer<float> halide_result = consumer.realize(160, 160);

        // Here's the equivalent (serial) C:
        float c_result[160][160];
        // For every strip of 16 scanlines (this loop is parallel in
        // the Halide version)
        for (int yo = 0; yo < 160/16 + 1; yo++) {

            // 16 doesn't divide 160, so push the last slice upwards
            // to fit within [0, 159] (see lesson 05).
            int y_base = yo * 16;
            if (y_base > 160-16) y_base = 160-16;

            // Allocate a two-scanline circular buffer for the producer
            float producer_storage[2][161];

            // For every scanline in the strip of 16:
            for (int yi = 0; yi < 16; yi++) {
                int y = y_base + yi;

                for (int py = y; py < y+2; py++) {
                    // Skip scanlines already computed *within this task*
                    if (yi > 0 && py == y) continue;

                    // Compute this scanline of the producer in 4-wide vectors
                    for (int x_vec = 0; x_vec < 160/4 + 1; x_vec++) {
                        int x_base = x_vec*4;
                        // 4 doesn't divide 161, so push the last vector left
                        // (see lesson 05).
                        if (x_base > 161 - 4) x_base = 161 - 4;
                        // If you're on x86, Halide generates SSE code for this part:
                        int x[] = {x_base, x_base + 1, x_base + 2, x_base + 3};
                        float vec[4] = {sinf(x[0] * py), sinf(x[1] * py),
                                        sinf(x[2] * py), sinf(x[3] * py)};
                        producer_storage[py & 1][x[0]] = vec[0];
                        producer_storage[py & 1][x[1]] = vec[1];
                        producer_storage[py & 1][x[2]] = vec[2];
                        producer_storage[py & 1][x[3]] = vec[3];
                    }
                }

                // Now compute consumer for this scanline:
                for (int x_vec = 0; x_vec < 160/4; x_vec++) {
                    int x_base = x_vec * 4;
                    // Again, Halide's equivalent here uses SSE.
                    int x[] = {x_base, x_base + 1, x_base + 2, x_base + 3};
                    float vec[] = {
                        (producer_storage[y & 1][x[0]] +
                         producer_storage[(y+1) & 1][x[0]] +
                         producer_storage[y & 1][x[0]+1] +
                         producer_storage[(y+1) & 1][x[0]+1])/4,
                        (producer_storage[y & 1][x[1]] +
                         producer_storage[(y+1) & 1][x[1]] +
                         producer_storage[y & 1][x[1]+1] +
                         producer_storage[(y+1) & 1][x[1]+1])/4,
                        (producer_storage[y & 1][x[2]] +
                         producer_storage[(y+1) & 1][x[2]] +
                         producer_storage[y & 1][x[2]+1] +
                         producer_storage[(y+1) & 1][x[2]+1])/4,
                        (producer_storage[y & 1][x[3]] +
                         producer_storage[(y+1) & 1][x[3]] +
                         producer_storage[y & 1][x[3]+1] +
                         producer_storage[(y+1) & 1][x[3]+1])/4,
                    };

                    c_result[y][x[0]] = vec[0];
                    c_result[y][x[1]] = vec[1];
                    c_result[y][x[2]] = vec[2];
                    c_result[y][x[3]] = vec[3];
                }

            }
        }
        printf("\nPseudo-code for the schedule:\n");
        consumer.print_loop_nest();

        // Let's check the C result against the Halide result. Doing
        // this I found several bugs in my C implementation, which
        // should tell you something.
        for (int y = 0; y < 160; y++) {
            for (int x = 0; x < 160; x++) {
                float error = halide_result(x, y) - c_result[y][x];
                // It's floating-point math, so we'll allow some slop:
                if (error < -0.001f || error > 0.001f) {
                    printf("halide_result(%d, %d) = %f instead of %f\n",
                           x, y, halide_result(x, y), c_result[y][x]);
                    return -1;
                }
            }
        }
    }

    printf("Success!\n");
    return 0;
}
