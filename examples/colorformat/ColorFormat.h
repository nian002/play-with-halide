/**
 * Copyright 2019 Xiaobin Wei <xiaobin.wee@gmail.com>
 */

#ifndef EXAMPLES_COLORFORMAT_H_
#define EXAMPLES_COLORFORMAT_H_

#include <stdint.h>
#include <stddef.h>


/**
 * We assume that BGR as the most common color format
 * So, all others use it for intermidiates
 */

///////////////////////////////////////////////////////////////
/// To BGR
///
void rgb2bgr(const uint8_t* src, uint8_t* dst, uint8_t width, uint8_t height);
void nv212bgr(const uint8_t* src, uint8_t* dst, uint8_t width, uint8_t height);
void nv122bgr(const uint8_t* src, uint8_t* dst, uint8_t width, uint8_t height);

///////////////////////////////////////////////////////////////
/// From BGR
///
void bgr2rgb(const uint8_t* src, uint8_t* dst, uint8_t width, uint8_t height);
void bgr2nv21(const uint8_t* src, uint8_t* dst, uint8_t width, uint8_t height);
void bgr2nv12(const uint8_t* src, uint8_t* dst, uint8_t width, uint8_t height);


#endif  // EXAMPLES_COLORFORMAT_H_
