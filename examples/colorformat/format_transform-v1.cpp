//  format_transform.cpp
//
//  Created by luofei on 2018/6/20.
//  Copyright © 2018年 Megvii. All rights reserved.
//


#include <opencv2/opencv.hpp>

#include "utils/meg_sdk_utils.h"
#include "utils/format_transform.h"
#include "meg_sdk_utils.h"
#define __ARM_NEON__ 1

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif


using namespace mghum;
using namespace cv;

//rgb to yuv (yuv to rgb) adopt full-range  criterion

const unsigned char low_index_arr[8] = {0, 0, 2, 2, 4, 4, 6, 6};
const unsigned char high_index_arr[8] = {1, 1 ,3, 3, 5, 5, 7, 7};


template <typename T>
inline T clamp(T x, T min, T max) {
    if (x < min)
        return min;
    if (x > max)
        return max;
    return x;
}


void FormatTransform::rgb2bgr(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height) {
    meg::ScopedTimer timer("rgb2bgr");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    for (int i = 0; i < image_height; ++i){
        for (int j = 0; j < image_width; ++j){
            int currentIndex = i * image_width + j;
            dst[currentIndex * 3 + 0] = src[currentIndex * 3 + 2];
            dst[currentIndex * 3 + 1] = src[currentIndex * 3 + 1];
            dst[currentIndex * 3 + 2] = src[currentIndex * 3 + 0];
        }
    }
    sdk_log("FormatTransform::rgb2bgr without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);
#else

    int count = image_width * image_height / 16;
    for (int i = 0; i < count ; ++i) {
        uint8x16x3_t bgr_register;
        uint8x16x3_t rgb_register = vld3q_u8(src);


        bgr_register.val[0] = rgb_register.val[2];
        bgr_register.val[1] = rgb_register.val[1];
        bgr_register.val[2] = rgb_register.val[0];

        vst3q_u8(dst, bgr_register);

        src += 16 * 3;
        dst += 16 * 3;
    }

    for (int i = count * 16; i < image_width * image_height ; ++i) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        dst += 3;
        src += 3;
    }
#endif
}
void FormatTransform::rgb2rgba(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height) {
    meg::ScopedTimer timer("rgb2rgba");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    for (int i = 0; i < image_height; ++i){
        for (int j = 0; j < image_width; ++j){

            int currentIndex = i * image_width + j;
            dst[currentIndex * 4 + 0] = src[currentIndex * 3 + 0];
            dst[currentIndex * 4 + 1] = src[currentIndex * 3 + 1];
            dst[currentIndex * 4 + 2] = src[currentIndex * 3 + 2];
            dst[currentIndex * 4 + 3] = 255;
        }
    }
    sdk_log("FormatTransform::rgb2rgba without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else

    uint8x16_t alpha = vdupq_n_u8(255);
    int count = image_width * image_height / 16;
    for (int i = 0; i < count ; ++i) {
        uint8x16x4_t rgba_register;
        uint8x16x3_t rgb_register = vld3q_u8(src);

        rgba_register.val[0] = rgb_register.val[0];
        rgba_register.val[1] = rgb_register.val[1];
        rgba_register.val[2] = rgb_register.val[2];
        rgba_register.val[3] = alpha;
        vst4q_u8(dst, rgba_register);

        src += 16 * 3;
        dst += 16 * 4;
    }
    for (int i = count * 16; i < image_width * image_height ; ++i) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst[3] = 255;
        dst += 4;
        src += 3;
    }
#endif
}
void FormatTransform::rgb2yuvnv12(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height){
    meg::ScopedTimer timer("rgb2yuvnv12");
#ifndef  __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    int nv21_len = image_width * image_height;
    int y, u, v;
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int curIndex = i * image_width + j;
            int currentR = src[curIndex * 3 + 0];
            int currentG = src[curIndex * 3 + 1];
            int currentB = src[curIndex * 3 + 2];

            y = ((77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8) + 0;
            u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
            v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;

            dst[i * image_width + j] = (uint8_t)clamp(y, 0 , 255);
            dst[nv21_len + (i >> 1) * image_width + (j & ~1) + 0] = (uint8_t)clamp(u, 0 ,255);
            dst[nv21_len + (i >> 1) * image_width + (j & ~1) + 1] = (uint8_t)clamp(v, 0 ,255);
        }
    }
    sdk_log("FormatTransform::rgb2yuvnv12 without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else

    int count = image_width / 16;
    int remainder = image_width % 16;

    const int16x8_t int16_rounding = vdupq_n_s16(128);

    const int16x8_t up_limit = vdupq_n_s16(255);
    const int16x8_t low_limit = vdupq_n_s16(0);

    int UVBeginIndex = image_width * image_height;
    uint8_t *uvdst = dst + UVBeginIndex;
    uint8_t *tempuv = new uint8_t[32];
    memset(tempuv, 0, 32);

    for (int i = 0; i < image_height; ++i)
    {
        if (i % 2 == 1)
        {
            for (int j = 0; j < count; ++j)
            {
                uint8x16x3_t rgb_register = vld3q_u8(src);

                uint8x16_t y_register;
                uint8x16x2_t uv_register;
                uint16x8_t high_r = vmovl_u8(vget_high_u8(rgb_register.val[0]));
                uint16x8_t low_r = vmovl_u8(vget_low_u8(rgb_register.val[0]));
                uint16x8_t high_g = vmovl_u8(vget_high_u8(rgb_register.val[1]));
                uint16x8_t low_g = vmovl_u8(vget_low_u8(rgb_register.val[1]));
                uint16x8_t high_b = vmovl_u8(vget_high_u8(rgb_register.val[2]));
                uint16x8_t low_b = vmovl_u8(vget_low_u8(rgb_register.val[2]));

                int16x8_t signed_high_r = vreinterpretq_s16_u16(high_r);
                int16x8_t signed_low_r = vreinterpretq_s16_u16(low_r);
                int16x8_t signed_high_g = vreinterpretq_s16_u16(high_g);
                int16x8_t signed_low_g = vreinterpretq_s16_u16(low_g);
                int16x8_t signed_high_b = vreinterpretq_s16_u16(high_b);
                int16x8_t signed_low_b = vreinterpretq_s16_u16(low_b);

                uint16x8_t high_y;
                uint16x8_t low_y;
                int16x8_t high_u;
                int16x8_t low_u;
                int16x8_t high_v;
                int16x8_t low_v;

                high_y = vmulq_u16(high_r, vdupq_n_u16(77));
                low_y = vmulq_u16(low_r ,vdupq_n_u16(77));

                high_u =vmulq_s16(signed_high_r, vdupq_n_s16(-43));
                low_u = vmulq_s16(signed_low_r, vdupq_n_s16(-43));

                high_v = vmulq_s16(signed_high_r, vdupq_n_s16(127));
                low_v = vmulq_s16(signed_low_r, vdupq_n_s16(127));

                ////
                high_y = vmlaq_u16(high_y, high_g ,vdupq_n_u16(150));
                low_y = vmlaq_u16(low_y, low_g, vdupq_n_u16(150));

                high_u = vmlaq_s16(high_u, signed_high_g, vdupq_n_s16(-84));
                low_u = vmlaq_s16(low_u, signed_low_g, vdupq_n_s16(-84));

                high_v = vmlaq_s16(high_v, signed_high_g, vdupq_n_s16(-106));
                low_v = vmlaq_s16(low_v, signed_low_g, vdupq_n_s16(-106));

                ////
                high_y = vmlaq_u16(high_y, high_b, vdupq_n_u16(29));
                low_y = vmlaq_u16(low_y, low_b, vdupq_n_u16(29));

                high_u = vmlaq_s16(high_u, signed_high_b, vdupq_n_s16(127));
                low_u = vmlaq_s16(low_u, signed_low_b, vdupq_n_s16(127));

                high_v = vmlaq_s16(high_v, signed_high_b, vdupq_n_s16(-21));
                low_v = vmlaq_s16(low_v, signed_low_b, vdupq_n_s16(-21));

                /// y_temp = y_temp + 128; u_temp = u_temp + 128; v_temp = v_temp + 128
                high_y = vaddq_u16(high_y, vdupq_n_u16(128));
                low_y = vaddq_u16(low_y, vdupq_n_u16(128));

                high_u = vaddq_s16(high_u, int16_rounding);
                low_u = vaddq_s16(low_u, int16_rounding);

                high_v = vaddq_s16(high_v, int16_rounding);
                low_v = vaddq_s16(low_v, int16_rounding);

                // y = y_temp >> 8; v= v_temp >> 8 + 128;  u = u_temp >> 8 + 128;

                uint8x16_t transformed_y;
                int8x16_t transformed_u;
                int8x16_t transformed_v;

                high_y = vaddq_u16(vshrq_n_u16(high_y, 8), vdupq_n_u16(0));
                high_u = vaddq_s16(vshrq_n_s16(high_u, 8), int16_rounding);
                high_v = vaddq_s16(vshrq_n_s16(high_v, 8), int16_rounding);

                low_y = vaddq_u16(vshrq_n_u16(low_y, 8), vdupq_n_u16(0));
                low_u = vaddq_s16(vshrq_n_s16(low_u, 8), int16_rounding);
                low_v = vaddq_s16(vshrq_n_s16(low_v, 8), int16_rounding);

                high_y = vminq_u16(vmaxq_u16(high_y, vdupq_n_u16(0)), vdupq_n_u16(255));
                high_u = vminq_s16(vmaxq_s16(high_u, low_limit), up_limit);
                high_v = vminq_s16(vmaxq_s16(high_v, low_limit), up_limit);

                low_y = vminq_u16(vmaxq_u16(low_y, vdupq_n_u16(0)), vdupq_n_u16(255));
                low_u = vminq_s16(vmaxq_s16(low_u, low_limit), up_limit);
                low_v = vminq_s16(vmaxq_s16(low_v, low_limit), up_limit);

                transformed_y = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y));
                transformed_u = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_u)), vqmovn_u16(vreinterpretq_u8_s16(high_u)));
                transformed_v = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_v)), vqmovn_u16(vreinterpretq_u8_s16(high_v)));

                y_register = transformed_y;
                uv_register.val[0] = vreinterpretq_u8_s8(transformed_u);
                uv_register.val[1] = vreinterpretq_u8_s8(transformed_v);

                vst1q_u8(dst, y_register);
                vst2q_u8(tempuv, uv_register);

                uint8x8x4_t targetuv = vld4_u8(tempuv);
                uint8x8x2_t final_uv_register;
                final_uv_register.val[0] = targetuv.val[2];
                final_uv_register.val[1] = targetuv.val[3];
                vst2_u8(uvdst, final_uv_register);

                src += 16 * 3;
                dst += 16;
                uvdst += 16;

            }
            for (int j = 0; j < remainder; ++j)
            {
                int currentR = src[0];
                int currentG = src[1];
                int currentB = src[2];
                int y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
                dst[0] = (uint8_t) clamp(y, 0, 255);
                if (j % 2 == 1)
                {
                    int u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
                    int v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;
                    uvdst[0] = (uint8_t) clamp(u, 0, 255);
                    uvdst[1] = (uint8_t) clamp(v, 0, 255);
                    uvdst += 2;
                }
                src += 3;
                dst++;
            }
        }
        else{
            for (int j = 0; j < count; ++j)
            {
                uint8x16x3_t bgr_register = vld3q_u8(src);
                uint8x16_t y_register;

                uint16x8_t signed_high_r = vmovl_u8(vget_high_u8(bgr_register.val[0]));
                uint16x8_t signed_low_r = vmovl_u8(vget_low_u8(bgr_register.val[0]));
                uint16x8_t signed_high_g = vmovl_u8(vget_high_u8(bgr_register.val[1]));
                uint16x8_t signed_low_g = vmovl_u8(vget_low_u8(bgr_register.val[1]));
                uint16x8_t signed_high_b = vmovl_u8(vget_high_u8(bgr_register.val[2]));
                uint16x8_t signed_low_b = vmovl_u8(vget_low_u8(bgr_register.val[2]));

                uint16x8_t high_y;
                uint16x8_t low_y;

                high_y = vmulq_u16(signed_high_r, vdupq_n_u16(77));
                low_y = vmulq_u16(signed_low_r ,vdupq_n_u16(77));
                high_y = vmlaq_u16(high_y, signed_high_g ,vdupq_n_u16(150));
                low_y = vmlaq_u16(low_y, signed_low_g, vdupq_n_u16(150));
                high_y = vmlaq_u16(high_y, signed_high_b, vdupq_n_u16(29));
                low_y = vmlaq_u16(low_y, signed_low_b, vdupq_n_u16(29));
                /// y_temp = y_temp + 128; u_temp = u_temp + 128; v_temp = v_temp + 128
                high_y = vaddq_u16(high_y, vdupq_n_u16(128));
                low_y = vaddq_u16(low_y, vdupq_n_u16(128));

                high_y = vshrq_n_u16(high_y, 8);
                low_y = vshrq_n_u16(low_y, 8);

                high_y = vminq_u16(vmaxq_u16(high_y, vdupq_n_u16(0)),  vdupq_n_u16(255));
                low_y = vminq_u16(vmaxq_u16(low_y,  vdupq_n_u16(0)),  vdupq_n_u16(255));

                // y = y_temp >> 8; v= v_temp >> 8 + 128;  u = u_temp >> 8 + 128;
                y_register = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y));

                vst1q_u8(dst, y_register);

                dst += 16;
                src += 16 * 3;
            }
            for (int j = 0; j < remainder; ++j)
            {
                int currentR = src[0];
                int currentG = src[1];
                int currentB = src[2];

                int y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
                dst[0] = (uint8_t)clamp(y, 0, 255);
                src += 3;
                dst++;
            }
        }
    }
    delete[](tempuv);
#endif
}
void FormatTransform::rgb2yuvnv21(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height){
    meg::ScopedTimer timer("rgb2yuvnv21");
#ifndef  __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    int nv21_len = image_width * image_height;
    int y, u, v;
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int curIndex = i * image_width + j;
            int currentR = src[curIndex * 3 + 0];
            int currentG = src[curIndex * 3 + 1];
            int currentB = src[curIndex * 3 + 2];

            y = ((77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8) + 0;
            u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
            v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;

            dst[i * image_width + j] = (uint8_t)clamp(y, 0, 255);
            dst[nv21_len + (i >> 1) * image_width + (j & ~1) + 0] = (uint8_t)clamp(v, 0, 255);
            dst[nv21_len + (i >> 1) * image_width + (j & ~1) + 1] = (uint8_t)clamp(u, 0, 255);
        }
    }

    sdk_log("FormatTransform::rgb2yuvnv21 without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else

    int count = image_width / 16;
    int remainder = image_width % 16;

    const int16x8_t int16_rounding = vdupq_n_s16(128);

    const int16x8_t up_limit = vdupq_n_s16(255);
    const int16x8_t low_limit = vdupq_n_s16(0);

    int UVBeginIndex = image_width * image_height;
    uint8_t *uvdst = dst + UVBeginIndex;
    uint8_t *tempuv = new uint8_t[32];
    memset(tempuv, 0, 32);

    for (int i = 0; i < image_height; ++i)
    {
        if (i % 2 == 1)
        {
            for (int j = 0; j < count; ++j)
            {
                uint8x16x3_t rgb_register = vld3q_u8(src);

                uint8x16_t y_register;
                uint8x16x2_t uv_register;
                uint16x8_t high_r = vmovl_u8(vget_high_u8(rgb_register.val[0]));
                uint16x8_t low_r = vmovl_u8(vget_low_u8(rgb_register.val[0]));
                uint16x8_t high_g = vmovl_u8(vget_high_u8(rgb_register.val[1]));
                uint16x8_t low_g = vmovl_u8(vget_low_u8(rgb_register.val[1]));
                uint16x8_t high_b = vmovl_u8(vget_high_u8(rgb_register.val[2]));
                uint16x8_t low_b = vmovl_u8(vget_low_u8(rgb_register.val[2]));

                int16x8_t signed_high_r = vreinterpretq_s16_u16(high_r);
                int16x8_t signed_low_r = vreinterpretq_s16_u16(low_r);
                int16x8_t signed_high_g = vreinterpretq_s16_u16(high_g);
                int16x8_t signed_low_g = vreinterpretq_s16_u16(low_g);
                int16x8_t signed_high_b = vreinterpretq_s16_u16(high_b);
                int16x8_t signed_low_b = vreinterpretq_s16_u16(low_b);

                uint16x8_t high_y;
                uint16x8_t low_y;
                int16x8_t high_u;
                int16x8_t low_u;
                int16x8_t high_v;
                int16x8_t low_v;

                high_y = vmulq_u16(high_r, vdupq_n_u16(77));
                low_y = vmulq_u16(low_r ,vdupq_n_u16(77));

                high_u =vmulq_s16(signed_high_r, vdupq_n_s16(-43));
                low_u = vmulq_s16(signed_low_r, vdupq_n_s16(-43));

                high_v = vmulq_s16(signed_high_r, vdupq_n_s16(127));
                low_v = vmulq_s16(signed_low_r, vdupq_n_s16(127));

                ////
                high_y = vmlaq_u16(high_y, high_g ,vdupq_n_u16(150));
                low_y = vmlaq_u16(low_y, low_g, vdupq_n_u16(150));

                high_u = vmlaq_s16(high_u, signed_high_g, vdupq_n_s16(-84));
                low_u = vmlaq_s16(low_u, signed_low_g, vdupq_n_s16(-84));

                high_v = vmlaq_s16(high_v, signed_high_g, vdupq_n_s16(-106));
                low_v = vmlaq_s16(low_v, signed_low_g, vdupq_n_s16(-106));

                ////
                high_y = vmlaq_u16(high_y, high_b, vdupq_n_u16(29));
                low_y = vmlaq_u16(low_y, low_b, vdupq_n_u16(29));

                high_u = vmlaq_s16(high_u, signed_high_b, vdupq_n_s16(127));
                low_u = vmlaq_s16(low_u, signed_low_b, vdupq_n_s16(127));

                high_v = vmlaq_s16(high_v, signed_high_b, vdupq_n_s16(-21));
                low_v = vmlaq_s16(low_v, signed_low_b, vdupq_n_s16(-21));

                /// y_temp = y_temp + 128; u_temp = u_temp + 128; v_temp = v_temp + 128
                high_y = vaddq_u16(high_y, vdupq_n_u16(128));
                low_y = vaddq_u16(low_y, vdupq_n_u16(128));

                high_u = vaddq_s16(high_u, int16_rounding);
                low_u = vaddq_s16(low_u, int16_rounding);

                high_v = vaddq_s16(high_v, int16_rounding);
                low_v = vaddq_s16(low_v, int16_rounding);

                // y = y_temp >> 8; v= v_temp >> 8 + 128;  u = u_temp >> 8 + 128;

                uint8x16_t transformed_y;
                int8x16_t transformed_u;
                int8x16_t transformed_v;

                high_y = vaddq_u16(vshrq_n_u16(high_y, 8), vdupq_n_u16(0));
                high_u = vaddq_s16(vshrq_n_s16(high_u, 8), int16_rounding);
                high_v = vaddq_s16(vshrq_n_s16(high_v, 8), int16_rounding);

                low_y = vaddq_u16(vshrq_n_u16(low_y, 8), vdupq_n_u16(0));
                low_u = vaddq_s16(vshrq_n_s16(low_u, 8), int16_rounding);
                low_v = vaddq_s16(vshrq_n_s16(low_v, 8), int16_rounding);

                high_y = vminq_u16(vmaxq_u16(high_y, vdupq_n_u16(0)), vdupq_n_u16(255));
                high_u = vminq_s16(vmaxq_s16(high_u, low_limit), up_limit);
                high_v = vminq_s16(vmaxq_s16(high_v, low_limit), up_limit);

                low_y = vminq_u16(vmaxq_u16(low_y, vdupq_n_u16(0)), vdupq_n_u16(255));
                low_u = vminq_s16(vmaxq_s16(low_u, low_limit), up_limit);
                low_v = vminq_s16(vmaxq_s16(low_v, low_limit), up_limit);

                transformed_y = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y));
                transformed_u = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_u)), vqmovn_u16(vreinterpretq_u8_s16(high_u)));
                transformed_v = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_v)), vqmovn_u16(vreinterpretq_u8_s16(high_v)));


                y_register = transformed_y;
                uv_register.val[0] = vreinterpretq_u8_s8(transformed_v);
                uv_register.val[1] = vreinterpretq_u8_s8(transformed_u);

                vst1q_u8(dst, y_register);
                vst2q_u8(tempuv, uv_register);

                uint8x8x4_t targetuv = vld4_u8(tempuv);
                uint8x8x2_t final_uv_register;
                final_uv_register.val[0] = targetuv.val[2];
                final_uv_register.val[1] = targetuv.val[3];
                vst2_u8(uvdst, final_uv_register);

                src += 16 * 3;
                dst += 16;
                uvdst += 16;

            }
            for (int j = 0; j < remainder; ++j)
            {
                int currentR = src[0];
                int currentG = src[1];
                int currentB = src[2];
                int y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
                dst[0] = (uint8_t)clamp(y, 0, 255);
                if (j % 2 == 1)
                {
                    int u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
                    int v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;
                    uvdst[0] = (uint8_t)clamp(v, 0, 255);
                    uvdst[1] = (uint8_t)clamp(u, 0, 255);
                    uvdst += 2;
                }
                src += 3;
                dst++;
            }
        }
        else{
            for (int j = 0; j < count; ++j)
            {
                uint8x16x3_t bgr_register = vld3q_u8(src);
                uint8x16_t y_register;

                uint16x8_t signed_high_r = vmovl_u8(vget_high_u8(bgr_register.val[0]));
                uint16x8_t signed_low_r = vmovl_u8(vget_low_u8(bgr_register.val[0]));
                uint16x8_t signed_high_g = vmovl_u8(vget_high_u8(bgr_register.val[1]));
                uint16x8_t signed_low_g = vmovl_u8(vget_low_u8(bgr_register.val[1]));
                uint16x8_t signed_high_b = vmovl_u8(vget_high_u8(bgr_register.val[2]));
                uint16x8_t signed_low_b = vmovl_u8(vget_low_u8(bgr_register.val[2]));

                uint16x8_t high_y;
                uint16x8_t low_y;

                high_y = vmulq_u16(signed_high_r, vdupq_n_u16(77));
                low_y = vmulq_u16(signed_low_r ,vdupq_n_u16(77));
                high_y = vmlaq_u16(high_y, signed_high_g ,vdupq_n_u16(150));
                low_y = vmlaq_u16(low_y, signed_low_g, vdupq_n_u16(150));
                high_y = vmlaq_u16(high_y, signed_high_b, vdupq_n_u16(29));
                low_y = vmlaq_u16(low_y, signed_low_b, vdupq_n_u16(29));
                /// y_temp = y_temp + 128; u_temp = u_temp + 128; v_temp = v_temp + 128
                high_y = vaddq_u16(high_y, vdupq_n_u16(128));
                low_y = vaddq_u16(low_y, vdupq_n_u16(128));

                high_y = vshrq_n_u16(high_y, 8);
                low_y = vshrq_n_u16(low_y, 8);

                high_y = vminq_u16(vmaxq_u16(high_y, vdupq_n_u16(0)),  vdupq_n_u16(255));
                low_y = vminq_u16(vmaxq_u16(low_y,  vdupq_n_u16(0)),  vdupq_n_u16(255));

                // y = y_temp >> 8; v= v_temp >> 8 + 128;  u = u_temp >> 8 + 128;
                y_register = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y));

                vst1q_u8(dst, y_register);
                dst += 16;
                src += 16 * 3;
            }
            for (int j = 0; j < remainder; ++j)
            {
                int currentR = src[0];
                int currentG = src[1];
                int currentB = src[2];

                int y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
                dst[0] = (uint8_t)clamp(y, 0, 255);
                src += 3;
                dst++;
            }
        }
    }
    delete[](tempuv);
#endif
}
void FormatTransform::rgb2fullyuv(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height){
    meg::ScopedTimer timer("rgb2fullyuv");
#ifndef  __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    int nv21_len = image_width * image_height;
    int y, u, v;
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int curIndex = i * image_width + j;

            int currentR = src[curIndex * 3 + 0];
            int currentG = src[curIndex * 3 + 1];
            int currentB = src[curIndex * 3 + 2];

            y = ((77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8) + 0;
            u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
            v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;
            dst[curIndex * 3 + 0] = (uint8_t)clamp(y, 0, 255);
            dst[curIndex * 3 + 1] = (uint8_t)clamp(u, 0, 255);
            dst[curIndex * 3 + 2] = (uint8_t)clamp(v, 0, 255);
        }
    }
    sdk_log("FormatTransform::rgb2fullyuv without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else

    int count = image_width / 16;
    int remainder = image_width % 16;

    const int16x8_t int16_rounding = vdupq_n_s16(128);
    const int16x8_t up_limit = vdupq_n_s16(255);
    const int16x8_t low_limit = vdupq_n_s16(0);

    for (int i = 0; i < image_height; ++i)
    {
        for (int j = 0; j < count; ++j)
        {
            uint8x16x3_t rgb_register = vld3q_u8(src);
            uint8x16x3_t yuv_register;

            uint16x8_t high_r = vmovl_u8(vget_high_u8(rgb_register.val[0]));
            uint16x8_t low_r = vmovl_u8(vget_low_u8(rgb_register.val[0]));
            uint16x8_t high_g = vmovl_u8(vget_high_u8(rgb_register.val[1]));
            uint16x8_t low_g = vmovl_u8(vget_low_u8(rgb_register.val[1]));
            uint16x8_t high_b = vmovl_u8(vget_high_u8(rgb_register.val[2]));
            uint16x8_t low_b = vmovl_u8(vget_low_u8(rgb_register.val[2]));

            int16x8_t signed_high_r = vreinterpretq_s16_u16(high_r);
            int16x8_t signed_low_r = vreinterpretq_s16_u16(low_r);
            int16x8_t signed_high_g = vreinterpretq_s16_u16(high_g);
            int16x8_t signed_low_g = vreinterpretq_s16_u16(low_g);
            int16x8_t signed_high_b = vreinterpretq_s16_u16(high_b);
            int16x8_t signed_low_b = vreinterpretq_s16_u16(low_b);

            uint16x8_t high_y;
            uint16x8_t low_y;
            int16x8_t high_u;
            int16x8_t low_u;
            int16x8_t high_v;
            int16x8_t low_v;

            high_y = vmulq_u16(high_r, vdupq_n_u16(77));
            low_y = vmulq_u16(low_r ,vdupq_n_u16(77));

            high_u =vmulq_s16(signed_high_r, vdupq_n_s16(-43));
            low_u = vmulq_s16(signed_low_r, vdupq_n_s16(-43));

            high_v = vmulq_s16(signed_high_r, vdupq_n_s16(127));
            low_v = vmulq_s16(signed_low_r, vdupq_n_s16(127));

            ////
            high_y = vmlaq_u16(high_y, high_g ,vdupq_n_u16(150));
            low_y = vmlaq_u16(low_y, low_g, vdupq_n_u16(150));

            high_u = vmlaq_s16(high_u, signed_high_g, vdupq_n_s16(-84));
            low_u = vmlaq_s16(low_u, signed_low_g, vdupq_n_s16(-84));

            high_v = vmlaq_s16(high_v, signed_high_g, vdupq_n_s16(-106));
            low_v = vmlaq_s16(low_v, signed_low_g, vdupq_n_s16(-106));

            ////
            high_y = vmlaq_u16(high_y, high_b, vdupq_n_u16(29));
            low_y = vmlaq_u16(low_y, low_b, vdupq_n_u16(29));

            high_u = vmlaq_s16(high_u, signed_high_b, vdupq_n_s16(127));
            low_u = vmlaq_s16(low_u, signed_low_b, vdupq_n_s16(127));

            high_v = vmlaq_s16(high_v, signed_high_b, vdupq_n_s16(-21));
            low_v = vmlaq_s16(low_v, signed_low_b, vdupq_n_s16(-21));

            /// y_temp = y_temp + 128; u_temp = u_temp + 128; v_temp = v_temp + 128
            high_y = vaddq_u16(high_y, vdupq_n_u16(128));
            low_y = vaddq_u16(low_y, vdupq_n_u16(128));

            high_u = vaddq_s16(high_u, int16_rounding);
            low_u = vaddq_s16(low_u, int16_rounding);

            high_v = vaddq_s16(high_v, int16_rounding);
            low_v = vaddq_s16(low_v, int16_rounding);

            high_y = vaddq_u16(vshrq_n_u16(high_y, 8), vdupq_n_u16(0));
            high_u = vaddq_s16(vshrq_n_s16(high_u, 8), int16_rounding);
            high_v = vaddq_s16(vshrq_n_s16(high_v, 8), int16_rounding);

            low_y = vaddq_u16(vshrq_n_u16(low_y, 8), vdupq_n_u16(0));
            low_u = vaddq_s16(vshrq_n_s16(low_u, 8), int16_rounding);
            low_v = vaddq_s16(vshrq_n_s16(low_v, 8), int16_rounding);

            high_y = vminq_u16(vmaxq_u16(high_y, vdupq_n_u16(0)), vdupq_n_u16(255));
            high_u = vminq_s16(vmaxq_s16(high_u, low_limit), up_limit);
            high_v = vminq_s16(vmaxq_s16(high_v, low_limit), up_limit);

            low_y = vminq_u16(vmaxq_u16(low_y, vdupq_n_u16(0)), vdupq_n_u16(255));
            low_u = vminq_s16(vmaxq_s16(low_u, low_limit), up_limit);
            low_v = vminq_s16(vmaxq_s16(low_v, low_limit), up_limit);

            uint8x16_t transformed_y;
            int8x16_t transformed_u;
            int8x16_t transformed_v;

            transformed_y = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y));
            transformed_u = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_u)), vqmovn_u16(vreinterpretq_u8_s16(high_u)));
            transformed_v = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_v)), vqmovn_u16(vreinterpretq_u8_s16(high_v)));

            yuv_register.val[0] = transformed_y;
            yuv_register.val[1] = vreinterpretq_u8_s8(transformed_u);
            yuv_register.val[2] = vreinterpretq_u8_s8(transformed_v);

            vst3q_u8(dst, yuv_register);

            src += 16 * 3;
            dst += 16 * 3;
        }
        for (int j = 0; j < remainder; ++j)
        {
            int currentR = src[0];
            int currentG = src[1];
            int currentB = src[2];
            int y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
            dst[0] = (uint8_t) clamp(y, 0, 255);

            int u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
            int v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;
            dst[1] = (uint8_t) clamp(u, 0, 255);
            dst[2] = (uint8_t) clamp(v, 0, 255);

            src += 3;
            dst +=3;
        }
    }
#endif
}

//rgba to ...
void FormatTransform::rgba2rgb(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width,  int image_height){
    meg::ScopedTimer timer("rgba2rgb");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    for (int i = 0; i < image_height; ++i){
        for (int j = 0; j < image_width; ++j){
            int currentIndex = i * image_width + j;
            dst[currentIndex * 3 + 0] = src[currentIndex * 4 + 0];
            dst[currentIndex * 3 + 1] = src[currentIndex * 4 + 1];
            dst[currentIndex * 3 + 2] = src[currentIndex * 4 + 2];
        }
    }
    sdk_log("FormatTransform::rgba2rgb without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else

    int count = image_width * image_height / 16;
    for (int i = 0; i < count ; ++i) {
        uint8x16x3_t bgr_register;
        uint8x16x4_t rgba_register = vld4q_u8(src);

        bgr_register.val[0] = rgba_register.val[0];
        bgr_register.val[1] = rgba_register.val[1];
        bgr_register.val[2] = rgba_register.val[2];

        vst3q_u8(dst, bgr_register);
        src += 16 * 4;
        dst += 16 * 3;
    }

    for (int i = count * 16; i < image_width * image_height ; ++i) {
        dst[0] = src[0];
        dst[1] = src[1];
        dst[2] = src[2];
        dst += 3;
        src += 4;
    }
#endif
}
void FormatTransform::rgba2bgr(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height) {
    meg::ScopedTimer timer("rgba2bgr");
#ifndef __ARM_NEON__
    for (int i = 0; i < image_height; ++i){
        for (int j = 0; j < image_width; ++j){
            int currentIndex = i * image_width + j;

            dst[currentIndex * 3 + 0] = src[currentIndex * 4 + 2];
            dst[currentIndex * 3 + 1] = src[currentIndex * 4 + 1];
            dst[currentIndex * 3 + 2] = src[currentIndex * 4 + 0];
        }
    }
#else

    int count = image_width * image_height / 16;
    for (int i = 0; i < count ; ++i) {
        uint8x16x3_t bgr_register;
        uint8x16x4_t rgba_register = vld4q_u8(src);

        bgr_register.val[0] = rgba_register.val[2];
        bgr_register.val[1] = rgba_register.val[1];
        bgr_register.val[2] = rgba_register.val[0];

        vst3q_u8(dst, bgr_register);
        src += 16 * 4;
        dst += 16 * 3;
    }

    for (int i = count * 16; i < image_width * image_height ; ++i) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        dst += 3;
        src += 4;
    }
#endif
}
void FormatTransform::rgba2yuvnv12(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height) {
    meg::ScopedTimer timer("rgba2yuvnv12");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    int y, u, v;
    int nv21_len = image_width * image_height;
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int curIndex = i * image_width + j;

            int currentR = src[curIndex * 4 + 0];
            int currentG = src[curIndex * 4 + 1];
            int currentB = src[curIndex * 4 + 2];

            y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
            u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
            v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;

            dst[i * image_width + j] =  (uint8_t)clamp(y, 0, 255);
            dst[nv21_len + (i >> 1) * image_width + (j & ~1) + 0] = (uint8_t)clamp(u, 0, 255);
            dst[nv21_len + (i >> 1) * image_width + (j & ~1) + 1] = (uint8_t)clamp(v, 0, 255);
        }
    }
    sdk_log("FormatTransform::rgba2yuvnv12 without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else
    int count = image_width / 16;
    int remainder = image_width % 16;

    const int16x8_t int16_rounding = vdupq_n_s16(128);
    const int16x8_t up_limit = vdupq_n_s16(255);
    const int16x8_t low_limit = vdupq_n_s16(0);

    int UVBeginIndex = image_width * image_height;
    uint8_t *uvdst = dst + UVBeginIndex;
    uint8_t *tempuv = new uint8_t[32];
    memset(tempuv, 0, 32);

    for (int i = 0; i < image_height; ++i)
    {
        if (i % 2 == 1)
        {
            for (int j = 0; j < count; ++j)
            {
                uint8x16x4_t rgba_register = vld4q_u8(src);
                uint8x16_t y_register;
                uint8x16x2_t uv_register;

                uint16x8_t high_r = vmovl_u8(vget_high_u8(rgba_register.val[0]));
                uint16x8_t low_r = vmovl_u8(vget_low_u8(rgba_register.val[0]));
                uint16x8_t high_g = vmovl_u8(vget_high_u8(rgba_register.val[1]));
                uint16x8_t low_g = vmovl_u8(vget_low_u8(rgba_register.val[1]));
                uint16x8_t high_b = vmovl_u8(vget_high_u8(rgba_register.val[2]));
                uint16x8_t low_b = vmovl_u8(vget_low_u8(rgba_register.val[2]));

                int16x8_t signed_high_r = vreinterpretq_s16_u16(high_r);
                int16x8_t signed_low_r = vreinterpretq_s16_u16(low_r);
                int16x8_t signed_high_g = vreinterpretq_s16_u16(high_g);
                int16x8_t signed_low_g = vreinterpretq_s16_u16(low_g);
                int16x8_t signed_high_b = vreinterpretq_s16_u16(high_b);
                int16x8_t signed_low_b = vreinterpretq_s16_u16(low_b);

                uint16x8_t high_y;
                uint16x8_t low_y;
                int16x8_t high_u;
                int16x8_t low_u;
                int16x8_t high_v;
                int16x8_t low_v;

                high_y = vmulq_u16(high_r, vdupq_n_u16(77));
                low_y = vmulq_u16(low_r ,vdupq_n_u16(77));

                high_u =vmulq_s16(signed_high_r, vdupq_n_s16(-43));
                low_u = vmulq_s16(signed_low_r, vdupq_n_s16(-43));

                high_v = vmulq_s16(signed_high_r, vdupq_n_s16(127));
                low_v = vmulq_s16(signed_low_r, vdupq_n_s16(127));

                ////
                high_y = vmlaq_u16(high_y, high_g ,vdupq_n_u16(150));
                low_y = vmlaq_u16(low_y, low_g, vdupq_n_u16(150));

                high_u = vmlaq_s16(high_u, signed_high_g, vdupq_n_s16(-84));
                low_u = vmlaq_s16(low_u, signed_low_g, vdupq_n_s16(-84));

                high_v = vmlaq_s16(high_v, signed_high_g, vdupq_n_s16(-106));
                low_v = vmlaq_s16(low_v, signed_low_g, vdupq_n_s16(-106));

                ////
                high_y = vmlaq_u16(high_y, high_b, vdupq_n_u16(29));
                low_y = vmlaq_u16(low_y, low_b, vdupq_n_u16(29));

                high_u = vmlaq_s16(high_u, signed_high_b, vdupq_n_s16(127));
                low_u = vmlaq_s16(low_u, signed_low_b, vdupq_n_s16(127));

                high_v = vmlaq_s16(high_v, signed_high_b, vdupq_n_s16(-21));
                low_v = vmlaq_s16(low_v, signed_low_b, vdupq_n_s16(-21));

                /// y_temp = y_temp + 128; u_temp = u_temp + 128; v_temp = v_temp + 128
                high_y = vaddq_u16(high_y, vdupq_n_u16(128));
                low_y = vaddq_u16(low_y, vdupq_n_u16(128));

                high_u = vaddq_s16(high_u, int16_rounding);
                low_u = vaddq_s16(low_u, int16_rounding);

                high_v = vaddq_s16(high_v, int16_rounding);
                low_v = vaddq_s16(low_v, int16_rounding);

                // y = y_temp >> 8; v= v_temp >> 8 + 128;  u = u_temp >> 8 + 128;

                uint8x16_t transformed_y;
                int8x16_t transformed_u;
                int8x16_t transformed_v;

                high_y = vaddq_u16(vshrq_n_u16(high_y, 8), vdupq_n_u16(0));
                high_u = vaddq_s16(vshrq_n_s16(high_u, 8), int16_rounding);
                high_v = vaddq_s16(vshrq_n_s16(high_v, 8), int16_rounding);

                low_y = vaddq_u16(vshrq_n_u16(low_y, 8), vdupq_n_u16(0));
                low_u = vaddq_s16(vshrq_n_s16(low_u, 8), int16_rounding);
                low_v = vaddq_s16(vshrq_n_s16(low_v, 8), int16_rounding);

                high_y = vminq_u16(vmaxq_u16(high_y, vdupq_n_u16(0)), vdupq_n_u16(255));
                high_u = vminq_s16(vmaxq_s16(high_u, low_limit), up_limit);
                high_v = vminq_s16(vmaxq_s16(high_v, low_limit), up_limit);

                low_y = vminq_u16(vmaxq_u16(low_y, vdupq_n_u16(0)), vdupq_n_u16(255));
                low_u = vminq_s16(vmaxq_s16(low_u, low_limit), up_limit);
                low_v = vminq_s16(vmaxq_s16(low_v, low_limit), up_limit);

                transformed_y = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y));
                transformed_u = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_u)), vqmovn_u16(vreinterpretq_u8_s16(high_u)));
                transformed_v = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_v)), vqmovn_u16(vreinterpretq_u8_s16(high_v)));

                y_register = transformed_y;
                uv_register.val[0] = vreinterpretq_u8_s8(transformed_u);
                uv_register.val[1] = vreinterpretq_u8_s8(transformed_v);

                vst1q_u8(dst, y_register);
                vst2q_u8(tempuv, uv_register);

                uint8x8x4_t targetuv = vld4_u8(tempuv);
                uint8x8x2_t final_uv_register;
                final_uv_register.val[0] = targetuv.val[2];
                final_uv_register.val[1] = targetuv.val[3];
                vst2_u8(uvdst, final_uv_register);

                src += 16 * 4;
                dst += 16;
                uvdst += 16;
            }
            for (int j = 0; j < remainder; ++j)
            {
                int currentR = src[0];
                int currentG = src[1];
                int currentB = src[2];
                int y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
                dst[0] = (uint8_t)clamp(y, 0, 255);
                if (j % 2 == 1)
                {
                    int u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
                    int v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;
                    uvdst[0] = (uint8_t)clamp(u, 0, 255);
                    uvdst[1] = (uint8_t)clamp(v, 0, 255);
                    uvdst += 2;
                }
                src += 4;
                dst++;
            }
        }
        else{
            for (int j = 0; j < count; ++j)
            {
                uint8x16x4_t bgr_register = vld4q_u8(src);
                uint8x16_t y_register;

                uint16x8_t signed_high_r = vmovl_u8(vget_high_u8(bgr_register.val[0]));
                uint16x8_t signed_low_r = vmovl_u8(vget_low_u8(bgr_register.val[0]));
                uint16x8_t signed_high_g = vmovl_u8(vget_high_u8(bgr_register.val[1]));
                uint16x8_t signed_low_g = vmovl_u8(vget_low_u8(bgr_register.val[1]));
                uint16x8_t signed_high_b = vmovl_u8(vget_high_u8(bgr_register.val[2]));
                uint16x8_t signed_low_b = vmovl_u8(vget_low_u8(bgr_register.val[2]));

                uint16x8_t high_y;
                uint16x8_t low_y;

                high_y = vmulq_u16(signed_high_r, vdupq_n_u16(77));
                low_y = vmulq_u16(signed_low_r ,vdupq_n_u16(77));
                high_y = vmlaq_u16(high_y, signed_high_g ,vdupq_n_u16(150));
                low_y = vmlaq_u16(low_y, signed_low_g, vdupq_n_u16(150));
                high_y = vmlaq_u16(high_y, signed_high_b, vdupq_n_u16(29));
                low_y = vmlaq_u16(low_y, signed_low_b, vdupq_n_u16(29));
                /// y_temp = y_temp + 128; u_temp = u_temp + 128; v_temp = v_temp + 128
                high_y = vaddq_u16(high_y, vdupq_n_u16(128));
                low_y = vaddq_u16(low_y, vdupq_n_u16(128));

                high_y = vshrq_n_u16(high_y, 8);
                low_y = vshrq_n_u16(low_y, 8);

                high_y = vminq_u16(vmaxq_u16(high_y, vdupq_n_u16(0)),  vdupq_n_u16(255));
                low_y = vminq_u16(vmaxq_u16(low_y,  vdupq_n_u16(0)),  vdupq_n_u16(255));

                // y = y_temp >> 8; v= v_temp >> 8 + 128;  u = u_temp >> 8 + 128;
                y_register = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y));

                vst1q_u8(dst, y_register);
                dst += 16;
                src += 16 * 4;
            }
            for (int j = 0; j < remainder; ++j)
            {
                int currentR = src[0];
                int currentG = src[1];
                int currentB = src[2];

                int y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
                dst[0] = (uint8_t)clamp(y, 0, 255);
                src += 4;
                dst++;
            }
        }
    }
    delete[](tempuv);
#endif
}
void FormatTransform::rgba2yuvnv21(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height) {
    meg::ScopedTimer timer("rgba2yuvnv21");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    int y, u, v;
    int nv21_len = image_width * image_height;
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int curIndex = i * image_width + j;

            int currentR = src[curIndex * 4 + 0];
            int currentG = src[curIndex * 4 + 1];
            int currentB = src[curIndex * 4 + 2];
            y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
            u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
            v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;
            dst[i * image_width + j] = (uint8_t)clamp(y, 0, 255);
            dst[nv21_len + (i >> 1) * image_width + (j & ~1) + 0] = (uint8_t)clamp(v, 0, 255);
            dst[nv21_len + (i >> 1) * image_width + (j & ~1) + 1] = (uint8_t)clamp(u, 0, 255);
        }
    }
    sdk_log("FormatTransform::rgba2yuvnv21 without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else
    int count = image_width / 16;
    int remainder = image_width % 16;

    const int16x8_t int16_rounding = vdupq_n_s16(128);
    const int16x8_t up_limit = vdupq_n_s16(255);
    const int16x8_t low_limit = vdupq_n_s16(0);

    int UVBeginIndex = image_width * image_height;
    uint8_t *uvdst = dst + UVBeginIndex;
    uint8_t *tempuv = new uint8_t[32];
    memset(tempuv, 0, 32);

    for (int i = 0; i < image_height; ++i)
    {
        if (i % 2 == 1)
        {
            for (int j = 0; j < count; ++j)
            {
                uint8x16x4_t rgba_register = vld4q_u8(src);
                uint8x16_t y_register;
                uint8x16x2_t uv_register;

                uint16x8_t high_r = vmovl_u8(vget_high_u8(rgba_register.val[0]));
                uint16x8_t low_r = vmovl_u8(vget_low_u8(rgba_register.val[0]));
                uint16x8_t high_g = vmovl_u8(vget_high_u8(rgba_register.val[1]));
                uint16x8_t low_g = vmovl_u8(vget_low_u8(rgba_register.val[1]));
                uint16x8_t high_b = vmovl_u8(vget_high_u8(rgba_register.val[2]));
                uint16x8_t low_b = vmovl_u8(vget_low_u8(rgba_register.val[2]));

                int16x8_t signed_high_r = vreinterpretq_s16_u16(high_r);
                int16x8_t signed_low_r = vreinterpretq_s16_u16(low_r);
                int16x8_t signed_high_g = vreinterpretq_s16_u16(high_g);
                int16x8_t signed_low_g = vreinterpretq_s16_u16(low_g);
                int16x8_t signed_high_b = vreinterpretq_s16_u16(high_b);
                int16x8_t signed_low_b = vreinterpretq_s16_u16(low_b);

                uint16x8_t high_y;
                uint16x8_t low_y;
                int16x8_t high_u;
                int16x8_t low_u;
                int16x8_t high_v;
                int16x8_t low_v;

                high_y = vmulq_u16(high_r, vdupq_n_u16(77));
                low_y = vmulq_u16(low_r ,vdupq_n_u16(77));

                high_u =vmulq_s16(signed_high_r, vdupq_n_s16(-43));
                low_u = vmulq_s16(signed_low_r, vdupq_n_s16(-43));

                high_v = vmulq_s16(signed_high_r, vdupq_n_s16(127));
                low_v = vmulq_s16(signed_low_r, vdupq_n_s16(127));

                ////
                high_y = vmlaq_u16(high_y, high_g ,vdupq_n_u16(150));
                low_y = vmlaq_u16(low_y, low_g, vdupq_n_u16(150));

                high_u = vmlaq_s16(high_u, signed_high_g, vdupq_n_s16(-84));
                low_u = vmlaq_s16(low_u, signed_low_g, vdupq_n_s16(-84));

                high_v = vmlaq_s16(high_v, signed_high_g, vdupq_n_s16(-106));
                low_v = vmlaq_s16(low_v, signed_low_g, vdupq_n_s16(-106));

                ////
                high_y = vmlaq_u16(high_y, high_b, vdupq_n_u16(29));
                low_y = vmlaq_u16(low_y, low_b, vdupq_n_u16(29));

                high_u = vmlaq_s16(high_u, signed_high_b, vdupq_n_s16(127));
                low_u = vmlaq_s16(low_u, signed_low_b, vdupq_n_s16(127));

                high_v = vmlaq_s16(high_v, signed_high_b, vdupq_n_s16(-21));
                low_v = vmlaq_s16(low_v, signed_low_b, vdupq_n_s16(-21));

                /// y_temp = y_temp + 128; u_temp = u_temp + 128; v_temp = v_temp + 128
                high_y = vaddq_u16(high_y, vdupq_n_u16(128));
                low_y = vaddq_u16(low_y, vdupq_n_u16(128));

                high_u = vaddq_s16(high_u, int16_rounding);
                low_u = vaddq_s16(low_u, int16_rounding);

                high_v = vaddq_s16(high_v, int16_rounding);
                low_v = vaddq_s16(low_v, int16_rounding);

                // y = y_temp >> 8; v= v_temp >> 8 + 128;  u = u_temp >> 8 + 128;

                uint8x16_t transformed_y;
                int8x16_t transformed_u;
                int8x16_t transformed_v;

                high_y = vaddq_u16(vshrq_n_u16(high_y, 8), vdupq_n_u16(0));
                high_u = vaddq_s16(vshrq_n_s16(high_u, 8), int16_rounding);
                high_v = vaddq_s16(vshrq_n_s16(high_v, 8), int16_rounding);

                low_y = vaddq_u16(vshrq_n_u16(low_y, 8), vdupq_n_u16(0));
                low_u = vaddq_s16(vshrq_n_s16(low_u, 8), int16_rounding);
                low_v = vaddq_s16(vshrq_n_s16(low_v, 8), int16_rounding);

                high_y = vminq_u16(vmaxq_u16(high_y, vdupq_n_u16(0)), vdupq_n_u16(255));
                high_u = vminq_s16(vmaxq_s16(high_u, low_limit), up_limit);
                high_v = vminq_s16(vmaxq_s16(high_v, low_limit), up_limit);

                low_y = vminq_u16(vmaxq_u16(low_y, vdupq_n_u16(0)), vdupq_n_u16(255));
                low_u = vminq_s16(vmaxq_s16(low_u, low_limit), up_limit);
                low_v = vminq_s16(vmaxq_s16(low_v, low_limit), up_limit);

                transformed_y = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y));
                transformed_u = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_u)), vqmovn_u16(vreinterpretq_u8_s16(high_u)));
                transformed_v = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_v)), vqmovn_u16(vreinterpretq_u8_s16(high_v)));

                y_register = transformed_y;
                uv_register.val[0] = vreinterpretq_u8_s8(transformed_v);
                uv_register.val[1] = vreinterpretq_u8_s8(transformed_u);

                vst1q_u8(dst, y_register);
                vst2q_u8(tempuv, uv_register);

                uint8x8x4_t targetuv = vld4_u8(tempuv);
                uint8x8x2_t final_uv_register;
                final_uv_register.val[0] = targetuv.val[2];
                final_uv_register.val[1] = targetuv.val[3];
                vst2_u8(uvdst, final_uv_register);

                src += 16 * 4;
                dst += 16;
                uvdst += 16;

            }
            for (int j = 0; j < remainder; ++j)
            {
                int currentR = src[0];
                int currentG = src[1];
                int currentB = src[2];
                int y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
                dst[0] = (uint8_t)clamp(y, 0, 255);
                if (j % 2 == 1)
                {
                    int u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
                    int v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;
                    uvdst[0] = (uint8_t)clamp(v, 0, 255);
                    uvdst[1] = (uint8_t)clamp(u, 0, 255);
                    uvdst += 2;
                }
                src += 4;
                dst++;
            }
        }
        else{
            for (int j = 0; j < count; ++j)
            {
                uint8x16x4_t rgba_register = vld4q_u8(src);
                uint8x16_t y_register;

                uint16x8_t signed_high_r = vmovl_u8(vget_high_u8(rgba_register.val[0]));
                uint16x8_t signed_low_r = vmovl_u8(vget_low_u8(rgba_register.val[0]));
                uint16x8_t signed_high_g = vmovl_u8(vget_high_u8(rgba_register.val[1]));
                uint16x8_t signed_low_g = vmovl_u8(vget_low_u8(rgba_register.val[1]));
                uint16x8_t signed_high_b = vmovl_u8(vget_high_u8(rgba_register.val[2]));
                uint16x8_t signed_low_b = vmovl_u8(vget_low_u8(rgba_register.val[2]));

                uint16x8_t high_y;
                uint16x8_t low_y;

                high_y = vmulq_u16(signed_high_r, vdupq_n_u16(77));
                low_y = vmulq_u16(signed_low_r ,vdupq_n_u16(77));
                high_y = vmlaq_u16(high_y, signed_high_g ,vdupq_n_u16(150));
                low_y = vmlaq_u16(low_y, signed_low_g, vdupq_n_u16(150));
                high_y = vmlaq_u16(high_y, signed_high_b, vdupq_n_u16(29));
                low_y = vmlaq_u16(low_y, signed_low_b, vdupq_n_u16(29));
                /// y_temp = y_temp + 128; u_temp = u_temp + 128; v_temp = v_temp + 128
                high_y = vaddq_u16(high_y, vdupq_n_u16(128));
                low_y = vaddq_u16(low_y, vdupq_n_u16(128));

                high_y = vshrq_n_u16(high_y, 8);
                low_y = vshrq_n_u16(low_y, 8);

                high_y = vminq_u16(vmaxq_u16(high_y, vdupq_n_u16(0)),  vdupq_n_u16(255));
                low_y = vminq_u16(vmaxq_u16(low_y,  vdupq_n_u16(0)),  vdupq_n_u16(255));

                // y = y_temp >> 8; v= v_temp >> 8 + 128;  u = u_temp >> 8 + 128;
                y_register = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y));

                vst1q_u8(dst, y_register);
                dst += 16;
                src += 16 * 4;
            }
            for (int j = 0; j < remainder; ++j)
            {
                int currentR = src[0];
                int currentG = src[1];
                int currentB = src[2];

                int y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
                dst[0] = (uint8_t)clamp(y, 0, 255);
                src += 4;
                dst++;
            }
        }
    }
    delete[](tempuv);
#endif
}
void FormatTransform::rgba2fullyuv(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height){
    meg::ScopedTimer timer("rgba2fullyuv");
#ifndef  __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    int nv21_len = image_width * image_height;
    int y, u, v;
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int curIndex = i * image_width + j;

            int currentR = src[curIndex * 4 + 0];
            int currentG = src[curIndex * 4 + 1];
            int currentB = src[curIndex * 4 + 2];

            y = ((77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8) + 0;
            u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
            v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;
            dst[curIndex * 3 + 0] = (uint8_t)clamp(y, 0, 255);
            dst[curIndex * 3 + 1] = (uint8_t)clamp(u, 0, 255);
            dst[curIndex * 3 + 2] = (uint8_t)clamp(v, 0, 255);
        }
    }
    sdk_log("FormatTransform::rgba2fullyuv without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else

    int count = image_width / 16;
    int remainder = image_width % 16;

    const int16x8_t int16_rounding = vdupq_n_s16(128);
    const int16x8_t up_limit = vdupq_n_s16(255);
    const int16x8_t low_limit = vdupq_n_s16(0);

    for (int i = 0; i < image_height; ++i)
    {
        for (int j = 0; j < count; ++j)
        {
            uint8x16x4_t rgba_register = vld4q_u8(src);
            uint8x16x3_t yuv_register;

            uint16x8_t high_r = vmovl_u8(vget_high_u8(rgba_register.val[0]));
            uint16x8_t low_r = vmovl_u8(vget_low_u8(rgba_register.val[0]));
            uint16x8_t high_g = vmovl_u8(vget_high_u8(rgba_register.val[1]));
            uint16x8_t low_g = vmovl_u8(vget_low_u8(rgba_register.val[1]));
            uint16x8_t high_b = vmovl_u8(vget_high_u8(rgba_register.val[2]));
            uint16x8_t low_b = vmovl_u8(vget_low_u8(rgba_register.val[2]));

            int16x8_t signed_high_r = vreinterpretq_s16_u16(high_r);
            int16x8_t signed_low_r = vreinterpretq_s16_u16(low_r);
            int16x8_t signed_high_g = vreinterpretq_s16_u16(high_g);
            int16x8_t signed_low_g = vreinterpretq_s16_u16(low_g);
            int16x8_t signed_high_b = vreinterpretq_s16_u16(high_b);
            int16x8_t signed_low_b = vreinterpretq_s16_u16(low_b);

            uint16x8_t high_y;
            uint16x8_t low_y;
            int16x8_t high_u;
            int16x8_t low_u;
            int16x8_t high_v;
            int16x8_t low_v;

            high_y = vmulq_u16(high_r, vdupq_n_u16(77));
            low_y = vmulq_u16(low_r ,vdupq_n_u16(77));

            high_u =vmulq_s16(signed_high_r, vdupq_n_s16(-43));
            low_u = vmulq_s16(signed_low_r, vdupq_n_s16(-43));

            high_v = vmulq_s16(signed_high_r, vdupq_n_s16(127));
            low_v = vmulq_s16(signed_low_r, vdupq_n_s16(127));

            ////
            high_y = vmlaq_u16(high_y, high_g ,vdupq_n_u16(150));
            low_y = vmlaq_u16(low_y, low_g, vdupq_n_u16(150));

            high_u = vmlaq_s16(high_u, signed_high_g, vdupq_n_s16(-84));
            low_u = vmlaq_s16(low_u, signed_low_g, vdupq_n_s16(-84));

            high_v = vmlaq_s16(high_v, signed_high_g, vdupq_n_s16(-106));
            low_v = vmlaq_s16(low_v, signed_low_g, vdupq_n_s16(-106));

            ////
            high_y = vmlaq_u16(high_y, high_b, vdupq_n_u16(29));
            low_y = vmlaq_u16(low_y, low_b, vdupq_n_u16(29));

            high_u = vmlaq_s16(high_u, signed_high_b, vdupq_n_s16(127));
            low_u = vmlaq_s16(low_u, signed_low_b, vdupq_n_s16(127));

            high_v = vmlaq_s16(high_v, signed_high_b, vdupq_n_s16(-21));
            low_v = vmlaq_s16(low_v, signed_low_b, vdupq_n_s16(-21));

            /// y_temp = y_temp + 128; u_temp = u_temp + 128; v_temp = v_temp + 128
            high_y = vaddq_u16(high_y, vdupq_n_u16(128));
            low_y = vaddq_u16(low_y, vdupq_n_u16(128));

            high_u = vaddq_s16(high_u, int16_rounding);
            low_u = vaddq_s16(low_u, int16_rounding);

            high_v = vaddq_s16(high_v, int16_rounding);
            low_v = vaddq_s16(low_v, int16_rounding);

            // y = y_temp >> 8; v= v_temp >> 8 + 128;  u = u_temp >> 8 + 128;

            uint8x16_t transformed_y;
            int8x16_t transformed_u;
            int8x16_t transformed_v;

            high_y = vaddq_u16(vshrq_n_u16(high_y, 8), vdupq_n_u16(0));
            high_u = vaddq_s16(vshrq_n_s16(high_u, 8), int16_rounding);
            high_v = vaddq_s16(vshrq_n_s16(high_v, 8), int16_rounding);

            low_y = vaddq_u16(vshrq_n_u16(low_y, 8), vdupq_n_u16(0));
            low_u = vaddq_s16(vshrq_n_s16(low_u, 8), int16_rounding);
            low_v = vaddq_s16(vshrq_n_s16(low_v, 8), int16_rounding);

            high_y = vminq_u16(vmaxq_u16(high_y, vdupq_n_u16(0)), vdupq_n_u16(255));
            high_u = vminq_s16(vmaxq_s16(high_u, low_limit), up_limit);
            high_v = vminq_s16(vmaxq_s16(high_v, low_limit), up_limit);

            low_y = vminq_u16(vmaxq_u16(low_y, vdupq_n_u16(0)), vdupq_n_u16(255));
            low_u = vminq_s16(vmaxq_s16(low_u, low_limit), up_limit);
            low_v = vminq_s16(vmaxq_s16(low_v, low_limit), up_limit);

            transformed_y = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y));
            transformed_u = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_u)), vqmovn_u16(vreinterpretq_u8_s16(high_u)));
            transformed_v = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_v)), vqmovn_u16(vreinterpretq_u8_s16(high_v)));

            yuv_register.val[0] = transformed_y;
            yuv_register.val[1] = vreinterpretq_u8_s8(transformed_u);
            yuv_register.val[2] = vreinterpretq_u8_s8(transformed_v);

            vst3q_u8(dst, yuv_register);

            dst += 16 * 3;
            src += 16 * 4;
        }
        for (int j = 0; j < remainder; ++j)
        {
            int currentR = src[0];
            int currentG = src[1];
            int currentB = src[2];
            int y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
            dst[0] = (uint8_t)clamp(y, 0, 255);

            int u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
            int v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;
            dst[1] = (uint8_t) clamp(u, 0, 255);
            dst[2] = (uint8_t) clamp(v, 0, 255);

            src += 4;
            dst +=3;
        }
    }
#endif
}


//bgr to ...
void FormatTransform::bgr2rgb(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height) {
    meg::ScopedTimer timer("bgr2rgb");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();

    for (int i = 0; i < image_height; ++i){
        for (int j = 0; j < image_width; ++j){
            int currentIndex = i * image_width + j;

            dst[currentIndex * 3 + 0] = src[currentIndex * 3 + 2];
            dst[currentIndex * 3 + 1] = src[currentIndex * 3 + 1];
            dst[currentIndex * 3 + 2] = src[currentIndex * 3 + 0];
        }
    }
    sdk_log("FormatTransform::bgr2rgb without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else
    //neon acceleration

    int count = image_width * image_height / 16;
    for (int i = 0; i < count ; ++i) {
        uint8x16x3_t rgb_register;
        uint8x16x3_t bgr_register = vld3q_u8(src);

        rgb_register.val[0] = bgr_register.val[2];
        rgb_register.val[1] = bgr_register.val[1];
        rgb_register.val[2] = bgr_register.val[0];

        vst3q_u8(dst, rgb_register);

        src += 16 * 3;
        dst += 16 * 3;
    }
    for (int i = count * 16; i < image_width * image_height ; ++i) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        dst += 3;
        src += 3;
    }
#endif
}
void FormatTransform::bgr2rgba(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height) {
    meg::ScopedTimer timer("bgr2rgba");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();

    for (int i = 0; i < image_height; ++i){
        for (int j = 0; j < image_width; ++j){
            int currentIndex = i * image_width + j;

            dst[currentIndex * 4 + 0] = src[currentIndex * 3 + 2];
            dst[currentIndex * 4 + 1] = src[currentIndex * 3 + 1];
            dst[currentIndex * 4 + 2] = src[currentIndex * 3 + 0];
            dst[currentIndex * 4 + 3] = 255;
        }
    }
    sdk_log("FormatTransform::bgr2rgba without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else
    //neon acceleration

    uint8x16_t uint8_alpha = vdupq_n_u8(255);
    int count = image_width * image_height / 16;
    for (int i = 0; i < count ; ++i) {
        uint8x16x4_t rgba_register;
        uint8x16x3_t bgr_register = vld3q_u8(src);

        rgba_register.val[0] = bgr_register.val[2];
        rgba_register.val[1] = bgr_register.val[1];
        rgba_register.val[2] = bgr_register.val[0];

        rgba_register.val[3] = uint8_alpha;

//
//        rgba_register.val[0] = uint8_alpha;
//        rgba_register.val[1] = bgr_register.val[0];
//        rgba_register.val[2] = bgr_register.val[1];
//        rgba_register.val[3] = bgr_register.val[2];

        vst4q_u8(dst, rgba_register);

        src += 16 * 3;
        dst += 16 * 4;
    }
    for (int i = count * 16; i < image_width * image_height ; ++i) {
        dst[0] = src[2];
        dst[1] = src[1];
        dst[2] = src[0];
        dst[3] = 255;
        dst += 4;
        src += 3;
    }
#endif
}
void FormatTransform::bgr2yuvnv12(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height) {
    meg::ScopedTimer timer("bgr2yuvnv12");
#ifndef  __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    int nv21_len = image_width * image_height;
    int y, u, v;
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int curIndex = i * image_width + j;
            int currentR = src[curIndex * 3 + 2];
            int currentG = src[curIndex * 3 + 1];
            int currentB = src[curIndex * 3 + 0];

            y = ((77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8) + 0;
            u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
            v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;

            dst[i * image_width + j] = (uint8_t)clamp(y, 0, 255);
            dst[nv21_len + (i >> 1) * image_width + (j & ~1) + 0] =  (uint8_t)clamp(u, 0, 255);
            dst[nv21_len + (i >> 1) * image_width + (j & ~1) + 1] =  (uint8_t)clamp(v, 0, 255);
        }
    }

    sdk_log("FormatTransform::bgr2yuvnv12 without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else

    int count = image_width / 16;
    int remainder = image_width % 16;

    const int16x8_t int16_rounding = vdupq_n_s16(128);

    const int16x8_t up_limit = vdupq_n_s16(255);
    const int16x8_t low_limit = vdupq_n_s16(0);

    int UVBeginIndex = image_width * image_height;
    uint8_t *uvdst = dst + UVBeginIndex;
    uint8_t *tempuv = new uint8_t[32];
    memset(tempuv, 0, 32);

    for (int i = 0; i < image_height; ++i)
    {
        if (i % 2 == 1)
        {
            for (int j = 0; j < count; ++j)
            {
                uint8x16x3_t bgr_register = vld3q_u8(src);
                uint8x16_t y_register;
                uint8x16x2_t uv_register;

                uint16x8_t high_b = vmovl_u8(vget_high_u8(bgr_register.val[0]));
                uint16x8_t low_b = vmovl_u8(vget_low_u8(bgr_register.val[0]));
                uint16x8_t high_g = vmovl_u8(vget_high_u8(bgr_register.val[1]));
                uint16x8_t low_g = vmovl_u8(vget_low_u8(bgr_register.val[1]));
                uint16x8_t high_r = vmovl_u8(vget_high_u8(bgr_register.val[2]));
                uint16x8_t low_r = vmovl_u8(vget_low_u8(bgr_register.val[2]));

                int16x8_t signed_high_r = vreinterpretq_s16_u16(high_r);
                int16x8_t signed_low_r = vreinterpretq_s16_u16(low_r);
                int16x8_t signed_high_g = vreinterpretq_s16_u16(high_g);
                int16x8_t signed_low_g = vreinterpretq_s16_u16(low_g);
                int16x8_t signed_high_b = vreinterpretq_s16_u16(high_b);
                int16x8_t signed_low_b = vreinterpretq_s16_u16(low_b);

                uint16x8_t high_y;
                uint16x8_t low_y;
                int16x8_t high_u;
                int16x8_t low_u;
                int16x8_t high_v;
                int16x8_t low_v;

                high_y = vmulq_u16(high_r, vdupq_n_u16(77));
                low_y = vmulq_u16(low_r ,vdupq_n_u16(77));

                high_u =vmulq_s16(signed_high_r, vdupq_n_s16(-43));
                low_u = vmulq_s16(signed_low_r, vdupq_n_s16(-43));

                high_v = vmulq_s16(signed_high_r, vdupq_n_s16(127));
                low_v = vmulq_s16(signed_low_r, vdupq_n_s16(127));

                ////
                high_y = vmlaq_u16(high_y, high_g ,vdupq_n_u16(150));
                low_y = vmlaq_u16(low_y, low_g, vdupq_n_u16(150));

                high_u = vmlaq_s16(high_u, signed_high_g, vdupq_n_s16(-84));
                low_u = vmlaq_s16(low_u, signed_low_g, vdupq_n_s16(-84));

                high_v = vmlaq_s16(high_v, signed_high_g, vdupq_n_s16(-106));
                low_v = vmlaq_s16(low_v, signed_low_g, vdupq_n_s16(-106));

                ////
                high_y = vmlaq_u16(high_y, high_b, vdupq_n_u16(29));
                low_y = vmlaq_u16(low_y, low_b, vdupq_n_u16(29));

                high_u = vmlaq_s16(high_u, signed_high_b, vdupq_n_s16(127));
                low_u = vmlaq_s16(low_u, signed_low_b, vdupq_n_s16(127));

                high_v = vmlaq_s16(high_v, signed_high_b, vdupq_n_s16(-21));
                low_v = vmlaq_s16(low_v, signed_low_b, vdupq_n_s16(-21));

                /// y_temp = y_temp + 128; u_temp = u_temp + 128; v_temp = v_temp + 128
                high_y = vaddq_u16(high_y, vdupq_n_u16(128));
                low_y = vaddq_u16(low_y, vdupq_n_u16(128));

                high_u = vaddq_s16(high_u, int16_rounding);
                low_u = vaddq_s16(low_u, int16_rounding);

                high_v = vaddq_s16(high_v, int16_rounding);
                low_v = vaddq_s16(low_v, int16_rounding);

                // y = y_temp >> 8; v= v_temp >> 8 + 128;  u = u_temp >> 8 + 128;

                uint8x16_t transformed_y;
                int8x16_t transformed_u;
                int8x16_t transformed_v;

                high_y = vaddq_u16(vshrq_n_u16(high_y, 8), vdupq_n_u16(0));
                high_u = vaddq_s16(vshrq_n_s16(high_u, 8), int16_rounding);
                high_v = vaddq_s16(vshrq_n_s16(high_v, 8), int16_rounding);

                low_y = vaddq_u16(vshrq_n_u16(low_y, 8), vdupq_n_u16(0));
                low_u = vaddq_s16(vshrq_n_s16(low_u, 8), int16_rounding);
                low_v = vaddq_s16(vshrq_n_s16(low_v, 8), int16_rounding);

                high_y = vminq_u16(vmaxq_u16(high_y, vdupq_n_u16(0)), vdupq_n_u16(255));
                high_u = vminq_s16(vmaxq_s16(high_u, low_limit), up_limit);
                high_v = vminq_s16(vmaxq_s16(high_v, low_limit), up_limit);

                low_y = vminq_u16(vmaxq_u16(low_y, vdupq_n_u16(0)), vdupq_n_u16(255));
                low_u = vminq_s16(vmaxq_s16(low_u, low_limit), up_limit);
                low_v = vminq_s16(vmaxq_s16(low_v, low_limit), up_limit);

                transformed_y = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y));
                transformed_u = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_u)), vqmovn_u16(vreinterpretq_u8_s16(high_u)));
                transformed_v = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_v)), vqmovn_u16(vreinterpretq_u8_s16(high_v)));

                y_register = transformed_y;
                uv_register.val[0] = vreinterpretq_u8_s8(transformed_u);
                uv_register.val[1] = vreinterpretq_u8_s8(transformed_v);

                vst1q_u8(dst, y_register);
                vst2q_u8(tempuv, uv_register);

                uint8x8x4_t targetuv = vld4_u8(tempuv);
                uint8x8x2_t final_uv_register;
                final_uv_register.val[0] = targetuv.val[2];
                final_uv_register.val[1] = targetuv.val[3];
                vst2_u8(uvdst, final_uv_register);

                src += 16 * 3;
                dst += 16;
                uvdst += 16;

            }
            for (int j = 0; j < remainder; ++j)
            {
                int currentR = src[2];
                int currentG = src[1];
                int currentB = src[0];
                int y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
                dst[0] = (uint8_t)clamp(y, 0, 255);
                if (j % 2 == 1)
                {
                    int u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
                    int v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;
                    uvdst[0] = (uint8_t)clamp(u, 0, 255);
                    uvdst[1] = (uint8_t)clamp(v, 0, 255);
                    uvdst += 2;
                }
                src += 3;
                dst++;
            }
        }
        else{
            for (int j = 0; j < count; ++j)
            {
                uint8x16x3_t bgr_register = vld3q_u8(src);
                uint8x16_t y_register;

                uint16x8_t signed_high_b = vmovl_u8(vget_high_u8(bgr_register.val[0]));
                uint16x8_t signed_low_b = vmovl_u8(vget_low_u8(bgr_register.val[0]));
                uint16x8_t signed_high_g = vmovl_u8(vget_high_u8(bgr_register.val[1]));
                uint16x8_t signed_low_g = vmovl_u8(vget_low_u8(bgr_register.val[1]));
                uint16x8_t signed_high_r = vmovl_u8(vget_high_u8(bgr_register.val[2]));
                uint16x8_t signed_low_r = vmovl_u8(vget_low_u8(bgr_register.val[2]));

                uint16x8_t high_y;
                uint16x8_t low_y;

                high_y = vmulq_u16(signed_high_r, vdupq_n_u16(77));
                low_y = vmulq_u16(signed_low_r ,vdupq_n_u16(77));
                high_y = vmlaq_u16(high_y, signed_high_g ,vdupq_n_u16(150));
                low_y = vmlaq_u16(low_y, signed_low_g, vdupq_n_u16(150));
                high_y = vmlaq_u16(high_y, signed_high_b, vdupq_n_u16(29));
                low_y = vmlaq_u16(low_y, signed_low_b, vdupq_n_u16(29));
                /// y_temp = y_temp + 128; u_temp = u_temp + 128; v_temp = v_temp + 128
                high_y = vaddq_u16(high_y, vdupq_n_u16(128));
                low_y = vaddq_u16(low_y, vdupq_n_u16(128));

                high_y = vshrq_n_u16(high_y, 8);
                low_y = vshrq_n_u16(low_y, 8);

                high_y = vminq_u16(vmaxq_u16(high_y, vdupq_n_u16(0)),  vdupq_n_u16(255));
                low_y = vminq_u16(vmaxq_u16(low_y,  vdupq_n_u16(0)),  vdupq_n_u16(255));

                // y = y_temp >> 8; v= v_temp >> 8 + 128;  u = u_temp >> 8 + 128;
                y_register = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y));

                vst1q_u8(dst, y_register);
                dst += 16;
                src += 16 * 3;
            }
            for (int j = 0; j < remainder; ++j)
            {
                int currentR = src[2];
                int currentG = src[1];
                int currentB = src[0];

                int y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
                dst[0] = (uint8_t)clamp(y, 0, 255);
                src += 3;
                dst++;
            }
        }
    }
    delete[](tempuv);

#endif
}
void FormatTransform::bgr2yuvnv21(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height) {
    meg::ScopedTimer timer("bgr2yuvnv21");
#ifndef  __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    int nv21_len = image_width * image_height;
    int y, u, v;
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int curIndex = i * image_width + j;
            int currentB = src[curIndex * 3 + 0];
            int currentG = src[curIndex * 3 + 1];
            int currentR = src[curIndex * 3 + 2];

            y = ((77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8) + 0;
            u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
            v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;

            dst[i * image_width + j] = (uint8_t)clamp(y, 0, 255);
            dst[nv21_len + (i >> 1) * image_width + (j & ~1) + 0] = (uint8_t)clamp(v, 0, 255);
            dst[nv21_len + (i >> 1) * image_width + (j & ~1) + 1] = (uint8_t)clamp(u, 0, 255);
        }
    }

    sdk_log("FormatTransform::bgr2yuvnv21 without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else

    int count = image_width / 16;
    int remainder = image_width % 16;

    const int16x8_t int16_rounding = vdupq_n_s16(128);
    const int16x8_t up_limit = vdupq_n_s16(255);
    const int16x8_t low_limit = vdupq_n_s16(0);


    int UVBeginIndex = image_width * image_height;
    uint8_t *uvdst = dst + UVBeginIndex;
    uint8_t *tempuv = new uint8_t[32];
    memset(tempuv, 0, 32);

    for (int i = 0; i < image_height; ++i)
    {
        if (i % 2 == 1)
        {
            for (int j = 0; j < count; ++j)
            {
                uint8x16x3_t bgr_register = vld3q_u8(src);
                uint8x16_t y_register;
                uint8x16x2_t uv_register;

                uint16x8_t high_b = vmovl_u8(vget_high_u8(bgr_register.val[0]));
                uint16x8_t low_b = vmovl_u8(vget_low_u8(bgr_register.val[0]));
                uint16x8_t high_g = vmovl_u8(vget_high_u8(bgr_register.val[1]));
                uint16x8_t low_g = vmovl_u8(vget_low_u8(bgr_register.val[1]));
                uint16x8_t high_r = vmovl_u8(vget_high_u8(bgr_register.val[2]));
                uint16x8_t low_r = vmovl_u8(vget_low_u8(bgr_register.val[2]));

                int16x8_t signed_high_r = vreinterpretq_s16_u16(high_r);
                int16x8_t signed_low_r = vreinterpretq_s16_u16(low_r);
                int16x8_t signed_high_g = vreinterpretq_s16_u16(high_g);
                int16x8_t signed_low_g = vreinterpretq_s16_u16(low_g);
                int16x8_t signed_high_b = vreinterpretq_s16_u16(high_b);
                int16x8_t signed_low_b = vreinterpretq_s16_u16(low_b);

                uint16x8_t high_y;
                uint16x8_t low_y;
                int16x8_t high_u;
                int16x8_t low_u;
                int16x8_t high_v;
                int16x8_t low_v;

                high_y = vmulq_u16(high_r, vdupq_n_u16(77));
                low_y = vmulq_u16(low_r ,vdupq_n_u16(77));

                high_u =vmulq_s16(signed_high_r, vdupq_n_s16(-43));
                low_u = vmulq_s16(signed_low_r, vdupq_n_s16(-43));

                high_v = vmulq_s16(signed_high_r, vdupq_n_s16(127));
                low_v = vmulq_s16(signed_low_r, vdupq_n_s16(127));

                ////
                high_y = vmlaq_u16(high_y, high_g ,vdupq_n_u16(150));
                low_y = vmlaq_u16(low_y, low_g, vdupq_n_u16(150));

                high_u = vmlaq_s16(high_u, signed_high_g, vdupq_n_s16(-84));
                low_u = vmlaq_s16(low_u, signed_low_g, vdupq_n_s16(-84));

                high_v = vmlaq_s16(high_v, signed_high_g, vdupq_n_s16(-106));
                low_v = vmlaq_s16(low_v, signed_low_g, vdupq_n_s16(-106));

                ////
                high_y = vmlaq_u16(high_y, high_b, vdupq_n_u16(29));
                low_y = vmlaq_u16(low_y, low_b, vdupq_n_u16(29));

                high_u = vmlaq_s16(high_u, signed_high_b, vdupq_n_s16(127));
                low_u = vmlaq_s16(low_u, signed_low_b, vdupq_n_s16(127));

                high_v = vmlaq_s16(high_v, signed_high_b, vdupq_n_s16(-21));
                low_v = vmlaq_s16(low_v, signed_low_b, vdupq_n_s16(-21));

                /// y_temp = y_temp + 128; u_temp = u_temp + 128; v_temp = v_temp + 128
                high_y = vaddq_u16(high_y, vdupq_n_u16(128));
                low_y = vaddq_u16(low_y, vdupq_n_u16(128));

                high_u = vaddq_s16(high_u, int16_rounding);
                low_u = vaddq_s16(low_u, int16_rounding);

                high_v = vaddq_s16(high_v, int16_rounding);
                low_v = vaddq_s16(low_v, int16_rounding);

                // y = y_temp >> 8; v= v_temp >> 8 + 128;  u = u_temp >> 8 + 128;

                uint8x16_t transformed_y;
                int8x16_t transformed_u;
                int8x16_t transformed_v;

                high_y = vaddq_u16(vshrq_n_u16(high_y, 8), vdupq_n_u16(0));
                high_u = vaddq_s16(vshrq_n_s16(high_u, 8), int16_rounding);
                high_v = vaddq_s16(vshrq_n_s16(high_v, 8), int16_rounding);

                low_y = vaddq_u16(vshrq_n_u16(low_y, 8), vdupq_n_u16(0));
                low_u = vaddq_s16(vshrq_n_s16(low_u, 8), int16_rounding);
                low_v = vaddq_s16(vshrq_n_s16(low_v, 8), int16_rounding);

                high_y = vminq_u16(vmaxq_u16(high_y, vdupq_n_u16(0)), vdupq_n_u16(255));
                high_u = vminq_s16(vmaxq_s16(high_u, low_limit), up_limit);
                high_v = vminq_s16(vmaxq_s16(high_v, low_limit), up_limit);

                low_y = vminq_u16(vmaxq_u16(low_y, vdupq_n_u16(0)), vdupq_n_u16(255));
                low_u = vminq_s16(vmaxq_s16(low_u, low_limit), up_limit);
                low_v = vminq_s16(vmaxq_s16(low_v, low_limit), up_limit);

                transformed_y = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y));
                transformed_u = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_u)), vqmovn_u16(vreinterpretq_u8_s16(high_u)));
                transformed_v = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_v)), vqmovn_u16(vreinterpretq_u8_s16(high_v)));

                y_register = transformed_y;
                uv_register.val[0] = vreinterpretq_u8_s8(transformed_v);
                uv_register.val[1] = vreinterpretq_u8_s8(transformed_u);

                vst1q_u8(dst, y_register);
                vst2q_u8(tempuv, uv_register);

                uint8x8x4_t targetuv = vld4_u8(tempuv);
                uint8x8x2_t final_uv_register;
                final_uv_register.val[0] = targetuv.val[2];
                final_uv_register.val[1] = targetuv.val[3];
                vst2_u8(uvdst, final_uv_register);

                src += 16 * 3;
                dst += 16;
                uvdst += 16;

            }
            for (int j = 0; j < remainder; ++j)
            {
                int currentR = src[2];
                int currentG = src[1];
                int currentB = src[0];
                int y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
                dst[0] = (uint8_t)clamp(y, 0, 255);
                if (j % 2 == 1)
                {
                    int u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
                    int v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;
                    uvdst[0] = (uint8_t)clamp(v, 0, 255);;
                    uvdst[1] = (uint8_t)clamp(u, 0, 255);;
                    uvdst += 2;
                }
                src += 3;
                dst++;
            }
        }
        else{
            for (int j = 0; j < count; ++j)
            {
                uint8x16x3_t bgr_register = vld3q_u8(src);
                uint8x16_t y_register;


                uint16x8_t signed_high_b = vmovl_u8(vget_high_u8(bgr_register.val[0]));
                uint16x8_t signed_low_b = vmovl_u8(vget_low_u8(bgr_register.val[0]));
                uint16x8_t signed_high_g = vmovl_u8(vget_high_u8(bgr_register.val[1]));
                uint16x8_t signed_low_g = vmovl_u8(vget_low_u8(bgr_register.val[1]));
                uint16x8_t signed_high_r = vmovl_u8(vget_high_u8(bgr_register.val[2]));
                uint16x8_t signed_low_r = vmovl_u8(vget_low_u8(bgr_register.val[2]));

                uint16x8_t high_y;
                uint16x8_t low_y;

                high_y = vmulq_u16(signed_high_r, vdupq_n_u16(77));
                low_y = vmulq_u16(signed_low_r ,vdupq_n_u16(77));
                high_y = vmlaq_u16(high_y, signed_high_g ,vdupq_n_u16(150));
                low_y = vmlaq_u16(low_y, signed_low_g, vdupq_n_u16(150));
                high_y = vmlaq_u16(high_y, signed_high_b, vdupq_n_u16(29));
                low_y = vmlaq_u16(low_y, signed_low_b, vdupq_n_u16(29));
                /// y_temp = y_temp + 128; u_temp = u_temp + 128; v_temp = v_temp + 128
                high_y = vaddq_u16(high_y, vdupq_n_u16(128));
                low_y = vaddq_u16(low_y, vdupq_n_u16(128));

                high_y = vshrq_n_u16(high_y, 8);
                low_y = vshrq_n_u16(low_y, 8);

                high_y = vminq_u16(vmaxq_u16(high_y, vdupq_n_u16(0)),  vdupq_n_u16(255));
                low_y = vminq_u16(vmaxq_u16(low_y,  vdupq_n_u16(0)),  vdupq_n_u16(255));

                // y = y_temp >> 8; v= v_temp >> 8 + 128;  u = u_temp >> 8 + 128;
                y_register = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y));

                vst1q_u8(dst, y_register);
                dst += 16;
                src += 16 * 3;
            }
            for (int j = 0; j < remainder; ++j)
            {
                int currentR = src[2];
                int currentG = src[1];
                int currentB = src[0];

                int y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
                dst[0] = (uint8_t)clamp(y, 0, 255);
                src += 3;
                dst++;
            }
        }
    }
    delete[](tempuv);
#endif
}
void FormatTransform::bgr2fullyuv(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height){
    meg::ScopedTimer timer("bgr2fullyuv");
#ifndef  __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    int nv21_len = image_width * image_height;
    int y, u, v;
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int curIndex = i * image_width + j;

            int currentR = src[curIndex * 3 + 2];
            int currentG = src[curIndex * 3 + 1];
            int currentB = src[curIndex * 3 + 0];

            y = ((77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8) + 0;
            u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
            v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;

            dst[curIndex * 3 + 0] = (uint8_t)clamp(y, 0, 255);
            dst[curIndex * 3 + 1] = (uint8_t)clamp(u, 0, 255);
            dst[curIndex * 3 + 2] = (uint8_t)clamp(v, 0, 255);
        }
    }

    sdk_log("FormatTransform::bgr2fullyuv without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else

    int count = image_width / 16;
    int remainder = image_width % 16;

    const int16x8_t int16_rounding = vdupq_n_s16(128);
    const int16x8_t up_limit = vdupq_n_s16(255);
    const int16x8_t low_limit = vdupq_n_s16(0);


    for (int i = 0; i < image_height; ++i)
    {
        for (int j = 0; j < count; ++j)
        {
            uint8x16x3_t bgr_register = vld3q_u8(src);
            uint8x16x3_t yuv_register;

            uint16x8_t high_b = vmovl_u8(vget_high_u8(bgr_register.val[0]));
            uint16x8_t low_b = vmovl_u8(vget_low_u8(bgr_register.val[0]));
            uint16x8_t high_g = vmovl_u8(vget_high_u8(bgr_register.val[1]));
            uint16x8_t low_g = vmovl_u8(vget_low_u8(bgr_register.val[1]));
            uint16x8_t high_r = vmovl_u8(vget_high_u8(bgr_register.val[2]));
            uint16x8_t low_r = vmovl_u8(vget_low_u8(bgr_register.val[2]));

            int16x8_t signed_high_r = vreinterpretq_s16_u16(high_r);
            int16x8_t signed_low_r = vreinterpretq_s16_u16(low_r);
            int16x8_t signed_high_g = vreinterpretq_s16_u16(high_g);
            int16x8_t signed_low_g = vreinterpretq_s16_u16(low_g);
            int16x8_t signed_high_b = vreinterpretq_s16_u16(high_b);
            int16x8_t signed_low_b = vreinterpretq_s16_u16(low_b);

            uint16x8_t high_y;
            uint16x8_t low_y;
            int16x8_t high_u;
            int16x8_t low_u;
            int16x8_t high_v;
            int16x8_t low_v;

            high_y = vmulq_u16(high_r, vdupq_n_u16(77));
            low_y = vmulq_u16(low_r ,vdupq_n_u16(77));

            high_u =vmulq_s16(signed_high_r, vdupq_n_s16(-43));
            low_u = vmulq_s16(signed_low_r, vdupq_n_s16(-43));

            high_v = vmulq_s16(signed_high_r, vdupq_n_s16(127));
            low_v = vmulq_s16(signed_low_r, vdupq_n_s16(127));

            ////
            high_y = vmlaq_u16(high_y, high_g ,vdupq_n_u16(150));
            low_y = vmlaq_u16(low_y, low_g, vdupq_n_u16(150));

            high_u = vmlaq_s16(high_u, signed_high_g, vdupq_n_s16(-84));
            low_u = vmlaq_s16(low_u, signed_low_g, vdupq_n_s16(-84));

            high_v = vmlaq_s16(high_v, signed_high_g, vdupq_n_s16(-106));
            low_v = vmlaq_s16(low_v, signed_low_g, vdupq_n_s16(-106));

            ////
            high_y = vmlaq_u16(high_y, high_b, vdupq_n_u16(29));
            low_y = vmlaq_u16(low_y, low_b, vdupq_n_u16(29));

            high_u = vmlaq_s16(high_u, signed_high_b, vdupq_n_s16(127));
            low_u = vmlaq_s16(low_u, signed_low_b, vdupq_n_s16(127));

            high_v = vmlaq_s16(high_v, signed_high_b, vdupq_n_s16(-21));
            low_v = vmlaq_s16(low_v, signed_low_b, vdupq_n_s16(-21));

            /// y_temp = y_temp + 128; u_temp = u_temp + 128; v_temp = v_temp + 128
            high_y = vaddq_u16(high_y, vdupq_n_u16(128));
            low_y = vaddq_u16(low_y, vdupq_n_u16(128));

            high_u = vaddq_s16(high_u, int16_rounding);
            low_u = vaddq_s16(low_u, int16_rounding);

            high_v = vaddq_s16(high_v, int16_rounding);
            low_v = vaddq_s16(low_v, int16_rounding);

            // y = y_temp >> 8; v= v_temp >> 8 + 128;  u = u_temp >> 8 + 128;

            uint8x16_t transformed_y;
            int8x16_t transformed_u;
            int8x16_t transformed_v;

            high_y = vaddq_u16(vshrq_n_u16(high_y, 8), vdupq_n_u16(0));
            high_u = vaddq_s16(vshrq_n_s16(high_u, 8), int16_rounding);
            high_v = vaddq_s16(vshrq_n_s16(high_v, 8), int16_rounding);

            low_y = vaddq_u16(vshrq_n_u16(low_y, 8), vdupq_n_u16(0));
            low_u = vaddq_s16(vshrq_n_s16(low_u, 8), int16_rounding);
            low_v = vaddq_s16(vshrq_n_s16(low_v, 8), int16_rounding);

            high_y = vminq_u16(vmaxq_u16(high_y, vdupq_n_u16(0)), vdupq_n_u16(255));
            high_u = vminq_s16(vmaxq_s16(high_u, low_limit), up_limit);
            high_v = vminq_s16(vmaxq_s16(high_v, low_limit), up_limit);

            low_y = vminq_u16(vmaxq_u16(low_y, vdupq_n_u16(0)), vdupq_n_u16(255));
            low_u = vminq_s16(vmaxq_s16(low_u, low_limit), up_limit);
            low_v = vminq_s16(vmaxq_s16(low_v, low_limit), up_limit);

            transformed_y = vcombine_u8(vqmovn_u16(low_y), vqmovn_u16(high_y));
            transformed_u = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_u)), vqmovn_u16(vreinterpretq_u8_s16(high_u)));
            transformed_v = vcombine_s8(vqmovn_u16(vreinterpretq_u8_s16(low_v)), vqmovn_u16(vreinterpretq_u8_s16(high_v)));

            yuv_register.val[0] = transformed_y;
            yuv_register.val[1] = vreinterpretq_u8_s8(transformed_u);
            yuv_register.val[2] = vreinterpretq_u8_s8(transformed_v);

            vst3q_u8(dst, yuv_register);

            src += 16 * 3;
            dst += 16 * 3;
        }
        for (int j = 0; j < remainder; ++j)
        {
            int currentR = src[0];
            int currentG = src[1];
            int currentB = src[2];
            int y = (77 * currentR + 150 * currentG + 29 * currentB + 128) >> 8;
            dst[0] = (uint8_t) clamp(y, 0, 255);

            int u = ((-43 * currentR - 84 * currentG + 127 * currentB + 128) >> 8) + 128;
            int v = ((127 * currentR - 106 * currentG - 21 * currentB + 128) >> 8) + 128;
            dst[1] = (uint8_t) clamp(u, 0, 255);
            dst[2] = (uint8_t) clamp(v, 0, 255);

            src += 3;
            dst +=3;
        }
    }
#endif
}

//nv12 to ...
void FormatTransform::yuvnv122rgb(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height){
    meg::ScopedTimer timer("yuvnv122rgb");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int index = i * image_width + j;
            int y = (int) src[index];
            int u = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 0];
            int v = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 1];

            int r = y + ((359 * (v - 128)) >> 8);
            int g = y - ((88 * (u - 128) + 183 * (v - 128)) >> 8);
            int b = y + ((454  * (u - 128)) >> 8);

            dst[index * 3 + 0] = (uint8_t)clamp(r, 0, 255);
            dst[index * 3 + 1] = (uint8_t)clamp(g, 0, 255);
            dst[index * 3 + 2] = (uint8_t)clamp(b, 0, 255);
        }
    }
    sdk_log("FormatTransform::yuvnv122rgb without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else
    int quotient = image_width / 8;
    int remainder = image_width % 8;

    const unsigned char *uv_src = src + image_width * image_height;

    const int32x4_t int32_minuend = vdupq_n_s32(-128);
    const int32x4_t up_limit = vdupq_n_s32(255);
    const int32x4_t low_limit = vdupq_n_s32(0);

    const uint8x8_t low_index_register = vld1_u8(low_index_arr);
    const uint8x8_t high_index_register = vld1_u8(high_index_arr);

    for (int i = 0; i < image_height; ++i) {
        for (int j = 0; j < quotient; ++j) {

            //dst register
            uint8x8x3_t rgb_register;
            uint8x8x3_t yuv_register;

            //temporary register
            uint8x8_t temp_uv_register;
            //load nv21 data into src register from memory
            yuv_register.val[0] = vld1_u8(src + i * image_width + 8 *j);
            temp_uv_register = vld1_u8(uv_src + i / 2 * image_width + 8 * j);

            yuv_register.val[1] = vtbl1_u8(temp_uv_register,low_index_register);
            yuv_register.val[2] = vtbl1_u8(temp_uv_register,high_index_register);

            int32x4_t signed_high_y = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[0]))));
            int32x4_t signed_low_y = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[0]))));
            int32x4_t signed_high_u = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[1]))));
            int32x4_t signed_low_u = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[1]))));
            int32x4_t signed_high_v = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[2]))));
            int32x4_t signed_low_v = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[2]))));

            int32x4_t high_r;
            int32x4_t low_r;
            int32x4_t high_g;
            int32x4_t low_g;
            int32x4_t high_b;
            int32x4_t low_b;

            high_r = vmulq_s32(vaddq_s32(signed_high_v, int32_minuend), vdupq_n_s32(359));
            high_g = vaddq_s32(vmulq_s32(vaddq_s32(signed_high_u, int32_minuend), vdupq_n_s32(-88)), vmulq_s32(vaddq_s32(signed_high_v, int32_minuend), vdupq_n_s32(-183)));
            high_b = vmulq_s32(vaddq_s32(signed_high_u, int32_minuend), vdupq_n_s32(454));

            low_r = vmulq_s32(vaddq_s32(signed_low_v, int32_minuend), vdupq_n_s32(359));
            low_g = vaddq_s32(vmulq_s32(vaddq_s32(signed_low_u, int32_minuend), vdupq_n_s32(-88)), vmulq_s32(vaddq_s32(signed_low_v, int32_minuend), vdupq_n_s32(-183)));
            low_b = vmulq_s32(vaddq_s32(signed_low_u, int32_minuend), vdupq_n_s32(454));

            high_r = vshrq_n_s32(high_r, 8);
            high_g = vshrq_n_s32(high_g, 8);
            high_b = vshrq_n_s32(high_b, 8);

            low_r = vshrq_n_s32(low_r, 8);
            low_g = vshrq_n_s32(low_g, 8);
            low_b = vshrq_n_s32(low_b, 8);

            high_r = vaddq_s32(high_r, signed_high_y);
            high_g = vaddq_s32(high_g, signed_high_y);
            high_b = vaddq_s32(high_b, signed_high_y);

            low_r = vaddq_s32(signed_low_y, low_r);
            low_g = vaddq_s32(signed_low_y, low_g);
            low_b = vaddq_s32(signed_low_y, low_b);

            high_r = vminq_s32(vmaxq_s32(high_r, low_limit), up_limit);
            high_g = vminq_s32(vmaxq_s32(high_g, low_limit), up_limit);
            high_b = vminq_s32(vmaxq_s32(high_b, low_limit), up_limit);

            low_r = vminq_s32(vmaxq_s32(low_r, low_limit), up_limit);
            low_g = vminq_s32(vmaxq_s32(low_g, low_limit), up_limit);
            low_b = vminq_s32(vmaxq_s32(low_b, low_limit), up_limit);


            rgb_register.val[0] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_r)), vqmovn_u32(vreinterpretq_u32_s32(high_r))));
            rgb_register.val[1] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_g)), vqmovn_u32(vreinterpretq_u32_s32(high_g))));
            rgb_register.val[2] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_b)), vqmovn_u32(vreinterpretq_u32_s32(high_b))));

            vst3_u8(dst, rgb_register);

            dst += 8*3;
        }
        for (int j = 0; j < remainder; ++j) {
            int y_index = i * image_width + 8 * quotient + j;
            int y = src[y_index];

            int uv_index =   i / 2 * image_width + 8 * quotient + j / 2;
            int u = uv_src[uv_index + 0];
            int v = uv_src[uv_index + 1];

            dst[0] = (uint8_t) clamp((y + ((359 * (v - 128)) >> 8)), 0, 255);
            dst[1] = (uint8_t) clamp((y - ((88 * (u - 128) + 183 * (v - 128)) >> 8)), 0, 255);
            dst[2] = (uint8_t) clamp((y + ((454 * (u - 128)) >> 8)), 0, 255);

            dst += 3;
        }
    }
#endif
}
void FormatTransform::yuvnv122rgba(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height){
    meg::ScopedTimer timer("yuvnv122rgba");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int index = i * image_width + j;
            int y = (int) src[index];
            int u = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 0];
            int v = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 1];

            int r = y + ((359 * (v - 128)) >> 8);
            int g = y - ((88 * (u - 128) + 183 * (v - 128)) >> 8);
            int b = y + ((454  * (u - 128)) >> 8);

            dst[index * 4 + 0] = (uint8_t)clamp(r, 0, 255);
            dst[index * 4 + 1] = (uint8_t)clamp(g, 0, 255);
            dst[index * 4 + 2] = (uint8_t)clamp(b, 0, 255);
            dst[index * 4 + 3] = 255;
        }
    }
    sdk_log("FormatTransform::yuvnv122rgba without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else
    int quotient = image_width / 8;
    int remainder = image_width % 8;

    const unsigned char *uv_src = src + image_width * image_height;

    const int32x4_t int32_minuend = vdupq_n_s32(-128);
    const int32x4_t up_limit = vdupq_n_s32(255);
    const int32x4_t low_limit = vdupq_n_s32(0);

    const uint8x8_t low_index_register = vld1_u8(low_index_arr);
    const uint8x8_t high_index_register = vld1_u8(high_index_arr);

    for (int i = 0; i < image_height; ++i) {
        for (int j = 0; j < quotient; ++j) {

            //dst register
            uint8x8x4_t rgba_register;
            uint8x8x3_t yuv_register;

            //temporary register
            uint8x8_t temp_uv_register;
            //load nv21 data into src register from memory
            yuv_register.val[0] = vld1_u8(src + i * image_width + 8 *j);
            temp_uv_register = vld1_u8(uv_src + i / 2 * image_width + 8 * j);

            yuv_register.val[1] = vtbl1_u8(temp_uv_register,low_index_register);
            yuv_register.val[2] = vtbl1_u8(temp_uv_register,high_index_register);

            int32x4_t signed_high_y = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[0]))));
            int32x4_t signed_low_y = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[0]))));
            int32x4_t signed_high_u = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[1]))));
            int32x4_t signed_low_u = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[1]))));
            int32x4_t signed_high_v = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[2]))));
            int32x4_t signed_low_v = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[2]))));

            int32x4_t high_r;
            int32x4_t low_r;
            int32x4_t high_g;
            int32x4_t low_g;
            int32x4_t high_b;
            int32x4_t low_b;

            high_r = vmulq_s32(vaddq_s32(signed_high_v, int32_minuend), vdupq_n_s32(359));
            high_g = vaddq_s32(vmulq_s32(vaddq_s32(signed_high_u, int32_minuend), vdupq_n_s32(-88)), vmulq_s32(vaddq_s32(signed_high_v, int32_minuend), vdupq_n_s32(-183)));
            high_b = vmulq_s32(vaddq_s32(signed_high_u, int32_minuend), vdupq_n_s32(454));

            low_r = vmulq_s32(vaddq_s32(signed_low_v, int32_minuend), vdupq_n_s32(359));
            low_g = vaddq_s32(vmulq_s32(vaddq_s32(signed_low_u, int32_minuend), vdupq_n_s32(-88)), vmulq_s32(vaddq_s32(signed_low_v, int32_minuend), vdupq_n_s32(-183)));
            low_b = vmulq_s32(vaddq_s32(signed_low_u, int32_minuend), vdupq_n_s32(454));

            high_r = vshrq_n_s32(high_r, 8);
            high_g = vshrq_n_s32(high_g, 8);
            high_b = vshrq_n_s32(high_b, 8);

            low_r = vshrq_n_s32(low_r, 8);
            low_g = vshrq_n_s32(low_g, 8);
            low_b = vshrq_n_s32(low_b, 8);

            high_r = vaddq_s32(high_r, signed_high_y);
            high_g = vaddq_s32(high_g, signed_high_y);
            high_b = vaddq_s32(high_b, signed_high_y);

            low_r = vaddq_s32(signed_low_y, low_r);
            low_g = vaddq_s32(signed_low_y, low_g);
            low_b = vaddq_s32(signed_low_y, low_b);

            high_r = vminq_s32(vmaxq_s32(high_r, low_limit), up_limit);
            high_g = vminq_s32(vmaxq_s32(high_g, low_limit), up_limit);
            high_b = vminq_s32(vmaxq_s32(high_b, low_limit), up_limit);

            low_r = vminq_s32(vmaxq_s32(low_r, low_limit), up_limit);
            low_g = vminq_s32(vmaxq_s32(low_g, low_limit), up_limit);
            low_b = vminq_s32(vmaxq_s32(low_b, low_limit), up_limit);


            rgba_register.val[0] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_r)), vqmovn_u32(vreinterpretq_u32_s32(high_r))));
            rgba_register.val[1] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_g)), vqmovn_u32(vreinterpretq_u32_s32(high_g))));
            rgba_register.val[2] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_b)), vqmovn_u32(vreinterpretq_u32_s32(high_b))));
            rgba_register.val[3] = vmov_n_u8(255);
            vst4_u8(dst, rgba_register);

            dst += 8*4;
        }
        for (int j = 0; j < remainder; ++j) {
            int y_index = i * image_width + 8 * quotient + j;
            int y = src[y_index];

            int uv_index =   i / 2 * image_width + 8 * quotient + j / 2;
            int u = uv_src[uv_index + 0];
            int v = uv_src[uv_index + 1];

            dst[0] = (uint8_t) clamp((y + ((359 * (v - 128)) >> 8)), 0, 255);
            dst[1] = (uint8_t) clamp((y - ((88 * (u - 128) + 183 * (v - 128)) >> 8)), 0, 255);
            dst[2] = (uint8_t) clamp((y + ((454 * (u - 128)) >> 8)), 0, 255);
            dst[3] =  255;
            dst += 4;
        }
    }

#endif
}
void FormatTransform::yuvnv122bgr(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height){
    meg::ScopedTimer timer("yuvnv122bgr");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int index = i * image_width + j;
            int y = (int) src[index];
            int u = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 0];
            int v = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 1];

            int r = y + ((359 * (v - 128)) >> 8);
            int g = y - ((88 * (u - 128) + 183 * (v - 128)) >> 8);
            int b = y + ((454  * (u - 128)) >> 8);

            dst[index * 3 + 0] = (uint8_t)clamp(b, 0, 255);
            dst[index * 3 + 1] = (uint8_t)clamp(g, 0, 255);
            dst[index * 3 + 2] = (uint8_t)clamp(r, 0, 255);
        }
    }
    sdk_log("FormatTransform::yuvnv122bgr without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else

    int quotient = image_width / 8;
    int remainder = image_width % 8;

    const unsigned char *uv_src = src + image_width * image_height;

    const int32x4_t int32_minuend = vdupq_n_s32(-128);
    const int32x4_t up_limit = vdupq_n_s32(255);
    const int32x4_t low_limit = vdupq_n_s32(0);

    const uint8x8_t low_index_register = vld1_u8(low_index_arr);
    const uint8x8_t high_index_register = vld1_u8(high_index_arr);

    for (int i = 0; i < image_height; ++i) {
        for (int j = 0; j < quotient; ++j) {

            //dst register
            uint8x8x3_t bgr_register;
            uint8x8x3_t yuv_register;

            //temporary register
            uint8x8_t temp_uv_register;
            //load nv21 data into src register from memory
            yuv_register.val[0] = vld1_u8(src + i * image_width + 8 *j);
            temp_uv_register = vld1_u8(uv_src + i / 2 * image_width + 8 * j);

            yuv_register.val[1] = vtbl1_u8(temp_uv_register,low_index_register);
            yuv_register.val[2] = vtbl1_u8(temp_uv_register,high_index_register);

            int32x4_t signed_high_y = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[0]))));
            int32x4_t signed_low_y = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[0]))));
            int32x4_t signed_high_u = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[1]))));
            int32x4_t signed_low_u = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[1]))));
            int32x4_t signed_high_v = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[2]))));
            int32x4_t signed_low_v = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[2]))));

            int32x4_t high_r;
            int32x4_t low_r;
            int32x4_t high_g;
            int32x4_t low_g;
            int32x4_t high_b;
            int32x4_t low_b;

            high_r = vmulq_s32(vaddq_s32(signed_high_v, int32_minuend), vdupq_n_s32(359));
            high_g = vaddq_s32(vmulq_s32(vaddq_s32(signed_high_u, int32_minuend), vdupq_n_s32(-88)), vmulq_s32(vaddq_s32(signed_high_v, int32_minuend), vdupq_n_s32(-183)));
            high_b = vmulq_s32(vaddq_s32(signed_high_u, int32_minuend), vdupq_n_s32(454));

            low_r = vmulq_s32(vaddq_s32(signed_low_v, int32_minuend), vdupq_n_s32(359));
            low_g = vaddq_s32(vmulq_s32(vaddq_s32(signed_low_u, int32_minuend), vdupq_n_s32(-88)), vmulq_s32(vaddq_s32(signed_low_v, int32_minuend), vdupq_n_s32(-183)));
            low_b = vmulq_s32(vaddq_s32(signed_low_u, int32_minuend), vdupq_n_s32(454));

            high_r = vshrq_n_s32(high_r, 8);
            high_g = vshrq_n_s32(high_g, 8);
            high_b = vshrq_n_s32(high_b, 8);

            low_r = vshrq_n_s32(low_r, 8);
            low_g = vshrq_n_s32(low_g, 8);
            low_b = vshrq_n_s32(low_b, 8);

            high_r = vaddq_s32(high_r, signed_high_y);
            high_g = vaddq_s32(high_g, signed_high_y);
            high_b = vaddq_s32(high_b, signed_high_y);

            low_r = vaddq_s32(signed_low_y, low_r);
            low_g = vaddq_s32(signed_low_y, low_g);
            low_b = vaddq_s32(signed_low_y, low_b);

            high_r = vminq_s32(vmaxq_s32(high_r, low_limit), up_limit);
            high_g = vminq_s32(vmaxq_s32(high_g, low_limit), up_limit);
            high_b = vminq_s32(vmaxq_s32(high_b, low_limit), up_limit);

            low_r = vminq_s32(vmaxq_s32(low_r, low_limit), up_limit);
            low_g = vminq_s32(vmaxq_s32(low_g, low_limit), up_limit);
            low_b = vminq_s32(vmaxq_s32(low_b, low_limit), up_limit);


            bgr_register.val[2] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_r)), vqmovn_u32(vreinterpretq_u32_s32(high_r))));
            bgr_register.val[1] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_g)), vqmovn_u32(vreinterpretq_u32_s32(high_g))));
            bgr_register.val[0] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_b)), vqmovn_u32(vreinterpretq_u32_s32(high_b))));

            vst3_u8(dst, bgr_register);

            dst += 8*3;
        }
        for (int j = 0; j < remainder; ++j) {
            int y_index = i * image_width + 8 * quotient + j;
            int y = src[y_index];

            int uv_index =   i / 2 * image_width + 8 * quotient + j / 2;
            int u = uv_src[uv_index + 0];
            int v = uv_src[uv_index + 1];

            dst[2] = (uint8_t) clamp((y + ((359 * (v - 128)) >> 8)), 0, 255);
            dst[1] = (uint8_t) clamp((y - ((88 * (u - 128) + 183 * (v - 128)) >> 8)), 0, 255);
            dst[0] = (uint8_t) clamp((y + ((454 * (u - 128)) >> 8)), 0, 255);

            dst += 3;
        }
    }

#endif
}
void FormatTransform::yuvnv122yuvnv21(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height) {
    memcpy(dst, src, image_width * image_height * sizeof(uint8_t));
    for (int i = image_height * image_width; i < image_width * image_height * 3 / 2; i += 2) {
        dst[i] = src[i + 1];
        dst[i + 1] = src[i];
    }
}
void FormatTransform::yuvnv122fullyuv(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height){
    meg::ScopedTimer timer("yuvnv122fullyuv");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int index = i * image_width + j;

            int y = (int) src[index];
            int u = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 0];
            int v = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 1];

            dst[index * 3 + 0] = (uint8_t)y;
            dst[index * 3 + 1] = (uint8_t)u;
            dst[index * 3 + 2] = (uint8_t)v;
        }
    }
    sdk_log("FormatTransform::yuvnv122fullyuv without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else

    int quotient = image_width / 16;
    int remainder = image_width % 16;

    const unsigned char *uv_src = src + image_width * image_height;

    uint8x8_t low_index_register = vld1_u8(low_index_arr);
    uint8x8_t high_index_register = vld1_u8(high_index_arr);


    for (int i = 0; i < image_height; ++i) {
        for (int j = 0; j < quotient; ++j) {

            //dst register
            uint8x16x3_t yuv_register;

            //temporary register
            uint8x8x2_t temp_uv_register;
            uint8x8x2_t u_register;
            uint8x8x2_t v_register;

            // src register
            uint8x16_t y_register;
            uint8x16x2_t uv_register;


            //load nv21 data into src register from memory
            y_register = vld1q_u8(src + i * image_width + 16 *j);
            temp_uv_register = vld2_u8(uv_src + i / 2 * image_width + 16 * j);

            u_register.val[0] = vtbl1_u8(temp_uv_register.val[0],low_index_register);
            u_register.val[1] = vtbl1_u8(temp_uv_register.val[0], high_index_register);

            v_register.val[0] = vtbl1_u8(temp_uv_register.val[1], low_index_register);
            v_register.val[1] = vtbl1_u8(temp_uv_register.val[1], high_index_register);

            uv_register.val[0] = vcombine_u8(u_register.val[0], u_register.val[1]);
            uv_register.val[1] = vcombine_u8(v_register.val[0], v_register.val[1]);

            yuv_register.val[0] = y_register;
            yuv_register.val[1] = uv_register.val[0];
            yuv_register.val[2] = uv_register.val[1];
            vst3q_u8(dst, yuv_register);

            dst += 16*3;
        }
        for (int j = 0; j < remainder; ++j) {
            int y_index = i * image_width + 16 * quotient + j;
            int uv_index =   i / 2 * image_width + 16 * quotient + j / 2;

            dst[0] = src[y_index];
            dst[1] = uv_src[uv_index + 0];
            dst[2] = uv_src[uv_index + 1];

            dst += 3;
        }
    }
#endif
}

//nv21 to ...
void FormatTransform::yuvnv212rgb(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height) {
    meg::ScopedTimer timer("yuvnv212rgb");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int index = i * image_width + j;
            int y = (int) src[index];
            int u = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 1];
            int v = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 0];

            int r = y + ((359 * (v - 128)) >> 8);
            int g = y - ((88 * (u - 128) + 183 * (v - 128)) >> 8);
            int b = y + ((454  * (u - 128)) >> 8);

            dst[index * 3 + 0] = (uint8_t)clamp(r, 0, 255);
            dst[index * 3 + 1] = (uint8_t)clamp(g, 0, 255);
            dst[index * 3 + 2] = (uint8_t)clamp(b, 0, 255);
        }
    }
    sdk_log("FormatTransform::yuvnv212rgb without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else
    int quotient = image_width / 8;
    int remainder = image_width % 8;

    const unsigned char *uv_src = src + image_width * image_height;

    const int32x4_t int32_minuend = vdupq_n_s32(-128);
    const int32x4_t up_limit = vdupq_n_s32(255);
    const int32x4_t low_limit = vdupq_n_s32(0);

    const uint8x8_t low_index_register = vld1_u8(low_index_arr);
    const uint8x8_t high_index_register = vld1_u8(high_index_arr);

    for (int i = 0; i < image_height; ++i) {
        for (int j = 0; j < quotient; ++j) {

            //dst register
            uint8x8x3_t rgb_register;
            uint8x8x3_t yuv_register;

            //temporary register
            uint8x8_t temp_uv_register;
            //load nv21 data into src register from memory
            yuv_register.val[0] = vld1_u8(src + i * image_width + 8 *j);
            temp_uv_register = vld1_u8(uv_src + i / 2 * image_width + 8 * j);

            yuv_register.val[2] = vtbl1_u8(temp_uv_register,low_index_register);
            yuv_register.val[1] = vtbl1_u8(temp_uv_register,high_index_register);

            int32x4_t signed_high_y = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[0]))));
            int32x4_t signed_low_y = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[0]))));
            int32x4_t signed_high_u = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[1]))));
            int32x4_t signed_low_u = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[1]))));
            int32x4_t signed_high_v = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[2]))));
            int32x4_t signed_low_v = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[2]))));

            int32x4_t high_r;
            int32x4_t low_r;
            int32x4_t high_g;
            int32x4_t low_g;
            int32x4_t high_b;
            int32x4_t low_b;

            high_r = vmulq_s32(vaddq_s32(signed_high_v, int32_minuend), vdupq_n_s32(359));
            high_g = vaddq_s32(vmulq_s32(vaddq_s32(signed_high_u, int32_minuend), vdupq_n_s32(-88)), vmulq_s32(vaddq_s32(signed_high_v, int32_minuend), vdupq_n_s32(-183)));
            high_b = vmulq_s32(vaddq_s32(signed_high_u, int32_minuend), vdupq_n_s32(454));

            low_r = vmulq_s32(vaddq_s32(signed_low_v, int32_minuend), vdupq_n_s32(359));
            low_g = vaddq_s32(vmulq_s32(vaddq_s32(signed_low_u, int32_minuend), vdupq_n_s32(-88)), vmulq_s32(vaddq_s32(signed_low_v, int32_minuend), vdupq_n_s32(-183)));
            low_b = vmulq_s32(vaddq_s32(signed_low_u, int32_minuend), vdupq_n_s32(454));

            high_r = vshrq_n_s32(high_r, 8);
            high_g = vshrq_n_s32(high_g, 8);
            high_b = vshrq_n_s32(high_b, 8);

            low_r = vshrq_n_s32(low_r, 8);
            low_g = vshrq_n_s32(low_g, 8);
            low_b = vshrq_n_s32(low_b, 8);

            high_r = vaddq_s32(high_r, signed_high_y);
            high_g = vaddq_s32(high_g, signed_high_y);
            high_b = vaddq_s32(high_b, signed_high_y);

            low_r = vaddq_s32(signed_low_y, low_r);
            low_g = vaddq_s32(signed_low_y, low_g);
            low_b = vaddq_s32(signed_low_y, low_b);

            high_r = vminq_s32(vmaxq_s32(high_r, low_limit), up_limit);
            high_g = vminq_s32(vmaxq_s32(high_g, low_limit), up_limit);
            high_b = vminq_s32(vmaxq_s32(high_b, low_limit), up_limit);

            low_r = vminq_s32(vmaxq_s32(low_r, low_limit), up_limit);
            low_g = vminq_s32(vmaxq_s32(low_g, low_limit), up_limit);
            low_b = vminq_s32(vmaxq_s32(low_b, low_limit), up_limit);


            rgb_register.val[0] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_r)), vqmovn_u32(vreinterpretq_u32_s32(high_r))));
            rgb_register.val[1] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_g)), vqmovn_u32(vreinterpretq_u32_s32(high_g))));
            rgb_register.val[2] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_b)), vqmovn_u32(vreinterpretq_u32_s32(high_b))));

            vst3_u8(dst, rgb_register);

            dst += 8*3;
        }
        for (int j = 0; j < remainder; ++j) {
            int y_index = i * image_width + 8 * quotient + j;
            int y = src[y_index];

            int uv_index =   i / 2 * image_width + 8 * quotient + j / 2;
            int u = uv_src[uv_index + 1];
            int v = uv_src[uv_index + 0];

            dst[0] = (uint8_t) clamp((y + ((359 * (v - 128)) >> 8)), 0, 255);
            dst[1] = (uint8_t) clamp((y - ((88 * (u - 128) + 183 * (v - 128)) >> 8)), 0, 255);
            dst[2] = (uint8_t) clamp((y + ((454 * (u - 128)) >> 8)), 0, 255);

            dst += 3;
        }
    }
#endif
}
void FormatTransform::yuvnv212rgba(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height) {
    meg::ScopedTimer timer("yuvnv212rgba");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int index = i * image_width + j;
            int y = (int) src[index];
            int u = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 1];
            int v = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 0];

            int r = y + ((359 * (v - 128)) >> 8);
            int g = y - ((88 * (u - 128) + 183 * (v - 128)) >> 8);
            int b = y + ((454  * (u - 128)) >> 8);

            dst[index * 4 + 0] = (uint8_t)clamp(r, 0, 255);
            dst[index * 4 + 1] = (uint8_t)clamp(g, 0, 255);
            dst[index * 4 + 2] = (uint8_t)clamp(b, 0, 255);
            dst[index * 4 + 3] = 255;

        }
    }
    sdk_log("FormatTransform::yuvnv212rgba without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else
    int quotient = image_width / 8;
    int remainder = image_width % 8;

    const unsigned char *uv_src = src + image_width * image_height;

    const int32x4_t int32_minuend = vdupq_n_s32(-128);
    const int32x4_t up_limit = vdupq_n_s32(255);
    const int32x4_t low_limit = vdupq_n_s32(0);

    const uint8x8_t low_index_register = vld1_u8(low_index_arr);
    const uint8x8_t high_index_register = vld1_u8(high_index_arr);

    for (int i = 0; i < image_height; ++i) {
        for (int j = 0; j < quotient; ++j) {

            //dst register
            uint8x8x4_t rgba_register;
            uint8x8x3_t yuv_register;

            //temporary register
            uint8x8_t temp_uv_register;
            //load nv21 data into src register from memory
            yuv_register.val[0] = vld1_u8(src + i * image_width + 8 *j);
            temp_uv_register = vld1_u8(uv_src + i / 2 * image_width + 8 * j);

            yuv_register.val[2] = vtbl1_u8(temp_uv_register,low_index_register);
            yuv_register.val[1] = vtbl1_u8(temp_uv_register,high_index_register);

            int32x4_t signed_high_y = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[0]))));
            int32x4_t signed_low_y = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[0]))));
            int32x4_t signed_high_u = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[1]))));
            int32x4_t signed_low_u = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[1]))));
            int32x4_t signed_high_v = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[2]))));
            int32x4_t signed_low_v = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[2]))));

            int32x4_t high_r;
            int32x4_t low_r;
            int32x4_t high_g;
            int32x4_t low_g;
            int32x4_t high_b;
            int32x4_t low_b;

            high_r = vmulq_s32(vaddq_s32(signed_high_v, int32_minuend), vdupq_n_s32(359));
            high_g = vaddq_s32(vmulq_s32(vaddq_s32(signed_high_u, int32_minuend), vdupq_n_s32(-88)), vmulq_s32(vaddq_s32(signed_high_v, int32_minuend), vdupq_n_s32(-183)));
            high_b = vmulq_s32(vaddq_s32(signed_high_u, int32_minuend), vdupq_n_s32(454));

            low_r = vmulq_s32(vaddq_s32(signed_low_v, int32_minuend), vdupq_n_s32(359));
            low_g = vaddq_s32(vmulq_s32(vaddq_s32(signed_low_u, int32_minuend), vdupq_n_s32(-88)), vmulq_s32(vaddq_s32(signed_low_v, int32_minuend), vdupq_n_s32(-183)));
            low_b = vmulq_s32(vaddq_s32(signed_low_u, int32_minuend), vdupq_n_s32(454));

            high_r = vshrq_n_s32(high_r, 8);
            high_g = vshrq_n_s32(high_g, 8);
            high_b = vshrq_n_s32(high_b, 8);

            low_r = vshrq_n_s32(low_r, 8);
            low_g = vshrq_n_s32(low_g, 8);
            low_b = vshrq_n_s32(low_b, 8);

            high_r = vaddq_s32(high_r, signed_high_y);
            high_g = vaddq_s32(high_g, signed_high_y);
            high_b = vaddq_s32(high_b, signed_high_y);

            low_r = vaddq_s32(signed_low_y, low_r);
            low_g = vaddq_s32(signed_low_y, low_g);
            low_b = vaddq_s32(signed_low_y, low_b);

            high_r = vminq_s32(vmaxq_s32(high_r, low_limit), up_limit);
            high_g = vminq_s32(vmaxq_s32(high_g, low_limit), up_limit);
            high_b = vminq_s32(vmaxq_s32(high_b, low_limit), up_limit);

            low_r = vminq_s32(vmaxq_s32(low_r, low_limit), up_limit);
            low_g = vminq_s32(vmaxq_s32(low_g, low_limit), up_limit);
            low_b = vminq_s32(vmaxq_s32(low_b, low_limit), up_limit);


            rgba_register.val[0] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_r)), vqmovn_u32(vreinterpretq_u32_s32(high_r))));
            rgba_register.val[1] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_g)), vqmovn_u32(vreinterpretq_u32_s32(high_g))));
            rgba_register.val[2] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_b)), vqmovn_u32(vreinterpretq_u32_s32(high_b))));
            rgba_register.val[3] = vmov_n_u8(255);
            vst4_u8(dst, rgba_register);

            dst += 8*4;
        }
        for (int j = 0; j < remainder; ++j) {
            int y_index = i * image_width + 8 * quotient + j;
            int y = src[y_index];

            int uv_index =   i / 2 * image_width + 8 * quotient + j / 2;
            int u = uv_src[uv_index + 1];
            int v = uv_src[uv_index + 0];

            dst[0] = (uint8_t) clamp((y + ((359 * (v - 128)) >> 8)), 0, 255);
            dst[1] = (uint8_t) clamp((y - ((88 * (u - 128) + 183 * (v - 128)) >> 8)), 0, 255);
            dst[2] = (uint8_t) clamp((y + ((454 * (u - 128)) >> 8)), 0, 255);
            dst[3] =  255;
            dst += 4;
        }
    }
#endif
}
void FormatTransform::yuvnv212bgr(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height) {
    meg::ScopedTimer timer("yuvnv212bgr");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int index = i * image_width + j;
            int y = (int) src[index];
            int u = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 1];
            int v = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 0];

            int r = y + ((359 * (v - 128)) >> 8);
            int g = y - ((88 * (u - 128) + 183 * (v - 128)) >> 8);
            int b = y + ((454  * (u - 128)) >> 8);

            dst[index * 3 + 0] = (uint8_t)clamp(b, 0, 255);
            dst[index * 3 + 1] = (uint8_t)clamp(g, 0, 255);
            dst[index * 3 + 2] = (uint8_t)clamp(r, 0, 255);
        }
    }
    sdk_log("FormatTransform::yuvnv212bgr without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else

    int quotient = image_width / 8;
    int remainder = image_width % 8;

    const unsigned char *uv_src = src + image_width * image_height;

    const int32x4_t int32_minuend = vdupq_n_s32(-128);
    const int32x4_t up_limit = vdupq_n_s32(255);
    const int32x4_t low_limit = vdupq_n_s32(0);

    const uint8x8_t low_index_register = vld1_u8(low_index_arr);
    const uint8x8_t high_index_register = vld1_u8(high_index_arr);

    for (int i = 0; i < image_height; ++i) {
        for (int j = 0; j < quotient; ++j) {

            //dst register
            uint8x8x3_t bgr_register;
            uint8x8x3_t yuv_register;

            //temporary register
            uint8x8_t temp_uv_register;
            //load nv21 data into src register from memory
            yuv_register.val[0] = vld1_u8(src + i * image_width + 8 *j);
            temp_uv_register = vld1_u8(uv_src + i / 2 * image_width + 8 * j);

            yuv_register.val[2] = vtbl1_u8(temp_uv_register,low_index_register);
            yuv_register.val[1] = vtbl1_u8(temp_uv_register,high_index_register);

            int32x4_t signed_high_y = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[0]))));
            int32x4_t signed_low_y = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[0]))));
            int32x4_t signed_high_u = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[1]))));
            int32x4_t signed_low_u = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[1]))));
            int32x4_t signed_high_v = vreinterpretq_s32_u32(vmovl_u16(vget_high_u16(vmovl_u8(yuv_register.val[2]))));
            int32x4_t signed_low_v = vreinterpretq_s32_u32(vmovl_u16(vget_low_u16(vmovl_u8(yuv_register.val[2]))));

            int32x4_t high_r;
            int32x4_t low_r;
            int32x4_t high_g;
            int32x4_t low_g;
            int32x4_t high_b;
            int32x4_t low_b;

            high_r = vmulq_s32(vaddq_s32(signed_high_v, int32_minuend), vdupq_n_s32(359));
            high_g = vaddq_s32(vmulq_s32(vaddq_s32(signed_high_u, int32_minuend), vdupq_n_s32(-88)), vmulq_s32(vaddq_s32(signed_high_v, int32_minuend), vdupq_n_s32(-183)));
            high_b = vmulq_s32(vaddq_s32(signed_high_u, int32_minuend), vdupq_n_s32(454));

            low_r = vmulq_s32(vaddq_s32(signed_low_v, int32_minuend), vdupq_n_s32(359));
            low_g = vaddq_s32(vmulq_s32(vaddq_s32(signed_low_u, int32_minuend), vdupq_n_s32(-88)), vmulq_s32(vaddq_s32(signed_low_v, int32_minuend), vdupq_n_s32(-183)));
            low_b = vmulq_s32(vaddq_s32(signed_low_u, int32_minuend), vdupq_n_s32(454));

            high_r = vshrq_n_s32(high_r, 8);
            high_g = vshrq_n_s32(high_g, 8);
            high_b = vshrq_n_s32(high_b, 8);

            low_r = vshrq_n_s32(low_r, 8);
            low_g = vshrq_n_s32(low_g, 8);
            low_b = vshrq_n_s32(low_b, 8);

            high_r = vaddq_s32(high_r, signed_high_y);
            high_g = vaddq_s32(high_g, signed_high_y);
            high_b = vaddq_s32(high_b, signed_high_y);

            low_r = vaddq_s32(signed_low_y, low_r);
            low_g = vaddq_s32(signed_low_y, low_g);
            low_b = vaddq_s32(signed_low_y, low_b);

            high_r = vminq_s32(vmaxq_s32(high_r, low_limit), up_limit);
            high_g = vminq_s32(vmaxq_s32(high_g, low_limit), up_limit);
            high_b = vminq_s32(vmaxq_s32(high_b, low_limit), up_limit);

            low_r = vminq_s32(vmaxq_s32(low_r, low_limit), up_limit);
            low_g = vminq_s32(vmaxq_s32(low_g, low_limit), up_limit);
            low_b = vminq_s32(vmaxq_s32(low_b, low_limit), up_limit);


            bgr_register.val[2] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_r)), vqmovn_u32(vreinterpretq_u32_s32(high_r))));
            bgr_register.val[1] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_g)), vqmovn_u32(vreinterpretq_u32_s32(high_g))));
            bgr_register.val[0] = vqmovn_u16(vcombine_u16(vqmovn_u32(vreinterpretq_u32_s32(low_b)), vqmovn_u32(vreinterpretq_u32_s32(high_b))));

            vst3_u8(dst, bgr_register);

            dst += 8*3;
        }
        for (int j = 0; j < remainder; ++j) {
            int y_index = i * image_width + 8 * quotient + j;
            int y = src[y_index];

            int uv_index =   i / 2 * image_width + 8 * quotient + j / 2;
            int u = uv_src[uv_index + 1];
            int v = uv_src[uv_index + 0];

            dst[2] = (uint8_t) clamp((y + ((359 * (v - 128)) >> 8)), 0, 255);
            dst[1] = (uint8_t) clamp((y - ((88 * (u - 128) + 183 * (v - 128)) >> 8)), 0, 255);
            dst[0] = (uint8_t) clamp((y + ((454 * (u - 128)) >> 8)), 0, 255);

            dst += 3;
        }
    }

#endif
}
void FormatTransform::yuvnv212yuvnv12(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height) {
    memcpy(dst, src, image_width * image_height * sizeof(uint8_t));
    for (int i = image_height * image_width; i < image_width * image_height * 3 / 2; i += 2) {
        dst[i] = src[i + 1];
        dst[i + 1] = src[i];
    }
}
void FormatTransform::yuvnv212fullyuv(const unsigned char * __restrict__ src, unsigned char * __restrict__ dst, int image_width, int image_height){
    meg::ScopedTimer timer("yuvnv212fullyuv");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int index = i * image_width + j;
            int y = (int) src[index];
            int u = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 1];
            int v = (int) src[image_height * image_width + (i / 2) * image_width + j / 2 * 2 + 0];

            dst[index * 3 + 0] = (uint8_t)y;
            dst[index * 3 + 1] = (uint8_t)u;
            dst[index * 3 + 2] = (uint8_t)v;
        }
    }
    sdk_log("FormatTransform::yuvnv212bgr without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else

    const int quotient = image_width / 16;
    const int remainder = image_width % 16;

    const unsigned char *uv_src = src + image_width * image_height;

    const uint8x8_t low_index_register = vld1_u8(low_index_arr);
    const uint8x8_t high_index_register = vld1_u8(high_index_arr);


    for (int i = 0; i < image_height; ++i) {
        for (int j = 0; j < quotient; ++j) {

            //dst register
            uint8x16x3_t yuv_register;

            //temporary register
            uint8x8x2_t temp_uv_register;
            uint8x8x2_t u_register;
            uint8x8x2_t v_register;

            // src register
            uint8x16_t y_register;
            uint8x16x2_t uv_register;


            //load nv21 data into src register from memory
            y_register = vld1q_u8(src + i * image_width + 16 *j);
            temp_uv_register = vld2_u8(uv_src + i / 2 * image_width + 16 * j);

            u_register.val[0] = vtbl1_u8(temp_uv_register.val[1],low_index_register);
            u_register.val[1] = vtbl1_u8(temp_uv_register.val[1], high_index_register);

            v_register.val[0] = vtbl1_u8(temp_uv_register.val[0], low_index_register);
            v_register.val[1] = vtbl1_u8(temp_uv_register.val[0], high_index_register);

            uv_register.val[0] = vcombine_u8(u_register.val[0], u_register.val[1]);
            uv_register.val[1] = vcombine_u8(v_register.val[0], v_register.val[1]);

            yuv_register.val[0] = y_register;
            yuv_register.val[1] = uv_register.val[0];
            yuv_register.val[2] = uv_register.val[1];
            vst3q_u8(dst, yuv_register);

            dst += 16*3;
        }
        for (int j = 0; j < remainder; ++j) {
            int y_index = i * image_width + 16 * quotient + j;
            int y = src[y_index];

            int uv_index =   i / 2 * image_width + 16 * quotient + j / 2;
            int u = uv_src[uv_index + 1];
            int v = uv_src[uv_index + 0];

            dst[0] = (uint8_t) y;
            dst[1] = (uint8_t) u;
            dst[2] = (uint8_t) v;

            dst += 3;
        }
    }

#endif
}


void FormatTransform::fullyuv2yuvnv21(const unsigned char * __restrict__ src, unsigned char * __restrict__ dst, int image_width, int image_height){
    meg::ScopedTimer timer("fullyuv2yuvnv21");
#ifndef __ARM_NEON__
    float normalBeginTime = (float)timer.GetMsecs();
    int nv21_len = image_width * image_height;
    int y, u, v;
    for (int i = 0; i < image_height; i++) {
        for (int j = 0; j < image_width; j++) {
            int curIndex = i * image_width + j;

            y = src[curIndex * 3 + 0];
            u = src[curIndex * 3 + 1];
            v = src[curIndex * 3 + 2];

            dst[i * image_width + j] = (uint8_t)clamp(y, 0, 255);
            dst[nv21_len + (i >> 1) * image_width + (j & ~1) + 0] = (uint8_t)clamp(v, 0, 255);
            dst[nv21_len + (i >> 1) * image_width + (j & ~1) + 1] = (uint8_t)clamp(u, 0, 255);
        }
    }
    sdk_log("FormatTransform::fullyuv2yuvnv21 without neon time(msec): %f", timer.GetMsecs() - normalBeginTime);

#else
    int count = image_width / 16;
    int remainder = image_width % 16;

    int UVBeginIndex = image_width * image_height;
    uint8_t *uvdst = dst + UVBeginIndex;
    const uint8_t temp_index[8] = {1, 3, 5, 7, 9, 11, 13, 15};

    for (int i = 0; i < image_height; ++i)
    {
        if (i % 2 == 1)
        {
            for (int j = 0; j < count; ++j)
            {
                uint8x16x3_t yuv_register = vld3q_u8(src);

                uint8x8x2_t u_register;
                uint8x8x2_t v_register;
                uint8x8_t index_register = vld1_u8(temp_index);

                u_register.val[0] =vget_low_u8(yuv_register.val[1]);
                u_register.val[1] =vget_high_u8(yuv_register.val[1]);

                v_register.val[0] =vget_low_u8(yuv_register.val[2]);
                v_register.val[1] =vget_high_u8(yuv_register.val[2]);


                uint8x8x2_t uv_register;
                uv_register.val[1] = vtbl2_u8(u_register, index_register);
                uv_register.val[0] = vtbl2_u8(v_register, index_register);

                vst1q_u8(dst, yuv_register.val[0]);
                vst2_u8(uvdst, uv_register);

                src += 16 * 3;
                dst += 16;
                uvdst += 16;

            }
            for (int j = 0; j < remainder; ++j)
            {
                dst[0] = src[0];
                if (j % 2 == 1)
                {
                    uvdst[0] = src[2]; //v
                    uvdst[1] = src[1]; //u
                    uvdst += 2;
                }
                src += 3;
                dst++;
            }
        }
        else{
            for (int j = 0; j < count; ++j)
            {
                uint8x16x3_t yuv_register = vld3q_u8(src);
                vst1q_u8(dst, yuv_register.val[0]);
                dst += 16;
                src += 16 * 3;
            }
            for (int j = 0; j < remainder; ++j)
            {
                dst[0] = src[0];
                src += 3;
                dst++;
            }
        }
    }
#endif
}

int clip(int x,int l,int h){
    if (x>h) x=h;
    if (x<l) x=l;
    return x;
}



void FormatTransform::yuvnv212yuvnv12_one_buff(unsigned char* __restrict__ src, int image_width, int image_height) {
    for (int i = image_height * image_width; i < image_height * 3 / 2 * image_width; i += 2) {
        unsigned char tmp = src[i];
        src[i] = src[i + 1];
        src[i + 1] = tmp;
    }
}

void FormatTransform::nv212resizedbgr(const unsigned char* src, int src_width, int src_height, unsigned char* dst, int dst_width, int dst_height) {
    for (int i = 0; i < dst_height; i++) {
        for (int j = 0; j < dst_width; j++) {
            int idx = i * dst_width + j;
            int ii = clamp(int(floor(((float) i + 0.5) / dst_height * src_height)), 0,
                           src_height - 1);
            int jj = clamp(int(floor(((float) j + 0.5) / dst_width * src_width)), 0, src_width - 1);
            int uv = src_height * src_width + ((ii / 2) * (src_width / 2) + (jj / 2)) * 2;
#ifndef __ARM_NEON__
            int Y = src[ii * src_width + jj], V = src[uv], U = src[uv + 1];
            dst[3 * idx + 2] = (unsigned char) clamp(((359 * (V - 128)) >> 8) + Y, 0, 255);
            dst[3 * idx + 1] = (unsigned char) clamp(-1 * ((88 * (U - 128) + 183 * (V - 128)) >> 8) + Y, 0, 255);
            dst[3 * idx] = (unsigned char) clamp(((454 * (U - 128)) >> 8) + Y, 0, 255);
#else
            uint8x8x3_t result;
            uint8x8_t y_ = vld1_u8(&src[ii * src_width + jj]);
            uint8x8x2_t uv_ = vld2_u8(&src[uv]);

            uint8x8_t temp1 = vdup_n_u8((uint8_t) 128);

            uv_.val[0] = vsub_u8(uv_.val[0], temp1);
            uv_.val[1] = vsub_u8(uv_.val[1], temp1);

            result.val[2] = vqadd_u8(vshr_n_u8(vmul_u8(vdup_n_u8((uint8_t) 359), uv_.val[0]), 8),
                                     vdup_n_u8((uint8_t) 0));
            result.val[1] = vqadd_u8(vsub_u8(y_, vshr_n_u8(
                    vadd_u8(vmul_u8(vdup_n_u8((uint8_t) 88), uv_.val[1]),
                            vmul_u8(vdup_n_u8((uint8_t) 183), uv_.val[0])), 8)), vdup_n_u8(0));
            result.val[0] = vqadd_u8(vadd_u8(y_, vshr_n_u8(
                                             vmul_u8(vdup_n_u8((uint8_t) 454), vsub_u8(uv_.val[1], temp1)), 8)),
                                     vdup_n_u8((uint8_t) 0));

            vst3_u8(&dst[3 * idx], result);
#endif
        }
    }
}

void FormatTransform::nv21resize(const unsigned char *src, int src_width, int src_height,
                                 unsigned char *dst, int dst_width, int dst_height) {
    /*
    Mat original(src_height + src_height / 2, src_width, CV_8UC1, (unsigned char *)src);
    Mat source(dst_height + dst_height / 2, dst_width, CV_8UC1, dst);
    resize(original, source, Size(dst_width, dst_height), (static_cast<void>(0), 0), (static_cast<void>(0), 0), cv::INTER_LINEAR);
*/
    {
        int sw = src_width;  //keyword is for local var to accelorate
        int sh = src_height;
        int dw = dst_width;
        int dh = dst_height;
        int y, x;
        unsigned long int srcy, srcx, src_index;// dst_index;
        unsigned long int xrIntFloat_16 = (sw << 16) / dw + 1; //better than float division
        unsigned long int yrIntFloat_16 = (sh << 16) / dh + 1;

        uint8_t* dst_uv = dst + dh * dw; //memory start pointer of dest uv
        uint8_t* src_uv = (unsigned char *)src + sh * sw; //memory start pointer of source uv
        uint8_t* dst_uv_yScanline;
        uint8_t* src_uv_yScanline;
        uint8_t* dst_y_slice = dst; //memory start pointer of dest y
        uint8_t* src_y_slice;
        uint8_t* sp;
        uint8_t* dp;

        for (y = 0; y < (dh & ~7); ++y)  //'dh & ~7' is to generate faster assembly code
        {
            srcy = (y * yrIntFloat_16) >> 16;
            src_y_slice = (unsigned char *)src + srcy * sw;

            if((y & 1) == 0)
            {
                dst_uv_yScanline = dst_uv + (y / 2) * dw;
                src_uv_yScanline = src_uv + (srcy / 2) * sw;
            }

            for(x = 0; x < (dw & ~7); ++x)
            {
                srcx = (x * xrIntFloat_16) >> 16;
                dst_y_slice[x] = src_y_slice[srcx];

                if((y & 1) == 0) //y is even
                {
                    if((x & 1) == 0) //x is even
                    {
                        src_index = (srcx / 2) * 2;

                        sp = dst_uv_yScanline + x;
                        dp = src_uv_yScanline + src_index;
                        *sp = *dp;
                        ++sp;
                        ++dp;
                        *sp = *dp;
                    }
                }
            }
            dst_y_slice += dw;
        }
    }
}

void FormatTransform::uint16ToFloat32(unsigned short *src, float *dst, int width, int height,
                                      unsigned short white,
                                      bool doBlackLevel, unsigned short black) {
    meg::ScopedTimer t("uint16ToFloat32");
    int srcLen = width * height;
    uint16x4_t srcRegister;
    float divisor;
    if(doBlackLevel){
        uint16x4_t blackLevel = vdup_n_u16(black);
        divisor = white-black;
        //float32x4_t divisorRegister = vdupq_n_f32(divisor);
        uint32x4_t tmp;
        float32x4_t ftmp;
#pragma omp parallel for
        for(int i=0;i<srcLen;i+=4){
            srcRegister = vld1_u16(src+i);
            srcRegister = vqsub_u16(srcRegister, blackLevel);
            tmp = vmovl_u16(srcRegister);
            ftmp = vcvtq_f32_u32(tmp);

            //ftmp = vdivq_f32(ftmp, divisorRegister);

            vst1q_f32(dst+i,ftmp);
        }
    }
    else{

        divisor = white;
        //float32x4_t divisorRegister = vdupq_n_f32(divisor);
        uint32x4_t tmp;
        float32x4_t ftmp;
#pragma omp parallel for
        for(int i=0;i<srcLen;i+=4){
            srcRegister = vld1_u16(src+i);
            tmp = vmovl_u16(srcRegister);
            ftmp = vcvtq_f32_u32(tmp);

            //ftmp = vdivq_f32(ftmp, divisorRegister);

            vst1q_f32(dst+i,ftmp);
        }
    }
    cv::Mat tmpmat(height, width, CV_32FC1, dst);
    tmpmat = tmpmat/divisor;

}

void FormatTransform::float32ToUint16(float *src, unsigned short *dst, int width, int height,
                                      unsigned short white, bool doBlackLevel, unsigned short black) {
    int srcLen = width * height;
    float32x4_t srcRegister;
    if(doBlackLevel){
        uint16x4_t blackLevel = vdup_n_u16(black);
        float multiplier = white-black;
        float32x4_t multiplierRegister = vdupq_n_f32(multiplier);
        uint32x4_t tmp;
        float32x4_t ftmp;
        uint16x4_t dstRegister;
#pragma omp parallel for
        for(int i=0;i<srcLen;i+=4){
            srcRegister = vld1q_f32(src+i);
            ftmp = vmulq_f32(srcRegister, multiplierRegister);
            ftmp = vminq_f32(multiplierRegister, ftmp);
            tmp = vcvtq_u32_f32(ftmp);
            dstRegister = vmovn_u32(tmp);
            dstRegister = vadd_u16(dstRegister, blackLevel);
            vst1_u16(dst+i, dstRegister);
        }
    }
    else{
        float multiplier = white;
        float32x4_t multiplierRegister = vdupq_n_f32(multiplier);
        uint32x4_t tmp;
        float32x4_t ftmp;
        uint16x4_t dstRegister;
#pragma omp parallel for
        for(int i=0;i<srcLen;i+=4){
            srcRegister = vld1q_f32(src+i);
            ftmp = vmulq_f32(srcRegister, multiplierRegister);
            ftmp = vminq_f32(multiplierRegister, ftmp);
            tmp = vcvtq_u32_f32(ftmp);
            dstRegister = vmovn_u32(tmp);
            vst1_u16(dst+i, dstRegister);
        }
    }
}


