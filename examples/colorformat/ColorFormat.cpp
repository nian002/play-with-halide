/**
 * Copyright 2019 Xiaobin Wei <xiaobin.wee@gmail.com>
 */

#include "ColorFormat.h"


void rgb2bgr(const uint8_t* src, uint8_t* dst, uint8_t width, uint8_t height) {
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int currentIndex = i * width + j;
            dst[currentIndex * 3 + 0] = src[currentIndex * 3 + 2];
            dst[currentIndex * 3 + 1] = src[currentIndex * 3 + 1];
            dst[currentIndex * 3 + 2] = src[currentIndex * 3 + 0];
        }
    }
}

void nv212bgr(const uint8_t* src, uint8_t* dst, uint8_t width, uint8_t height) {

}

void nv122bgr(const uint8_t* src, uint8_t* dst, uint8_t width, uint8_t height) {

}

void bgr2rgb(const uint8_t* src, uint8_t* dst, uint8_t width, uint8_t height) {
    rgb2bgr(src, dst, width, height);
}