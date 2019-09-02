//  format_transform.h
//
//  Created by LiangJuzi on 2018/1/23.
//  Copyright © 2018年 Megvii. All rights reserved.
//

#ifndef HUMANEFFECT_UTILS_FORMATTRANSFORM_H
#define HUMANEFFECT_UTILS_FORMATTRANSFORM_H

#include <stdio.h>

namespace mghum {
    class FormatTransform {
    public:
        /**
         * image format transform public param.
         * @param src[input] original image data buffer
         * @param image_width[input] original image width
         * @param image_height[input] original image height
         * @param dst[output] export image data buffer
         */
        static void rgb2bgr(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void rgb2rgba(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void rgb2yuvnv12(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void rgb2yuvnv21(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void rgb2fullyuv(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);


        static void rgba2rgb(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void rgba2bgr(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void rgba2yuvnv12(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void rgba2yuvnv21(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void rgba2fullyuv(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);

        static void bgr2rgb(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void bgr2rgba(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void bgr2yuvnv12(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void bgr2yuvnv21(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void bgr2fullyuv(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);

        static void yuvnv122rgb(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void yuvnv122rgba(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void yuvnv122bgr(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void yuvnv122yuvnv21(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void yuvnv122fullyuv(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);


        static void yuvnv212rgb(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void yuvnv212rgba(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void yuvnv212bgr(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void yuvnv212yuvnv12(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);
        static void yuvnv212fullyuv(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);


        static void fullyuv2yuvnv21(const unsigned char* __restrict__ src, unsigned char* __restrict__ dst, int image_width, int image_height);


        /**
         * image format yuvnv12 and yuvnv21 mutual transform.
         * src is yuvnv21, dst is yuvnv12.
         * src is yuvnv12, dst is yuvnv21.
         */

        static void yuvnv212yuvnv12_one_buff(unsigned char* __restrict__ src, int image_width, int image_height);

        /**
         * resize and transform
         * @param src
         * @param src_width
         * @param src_height
         * @param dst
         * @param dst_width  must align to 4
         * @param dst_height must align to 4
         */
        static void nv212resizedbgr(const unsigned char* src, int src_width, int src_height, unsigned char* dst, int dst_width, int dst_height);

        static void nv21resize(const unsigned char* src, int src_width, int src_height,
                               unsigned char* dst, int dst_width, int dst_height);

        static void uint16ToFloat32(unsigned short *src, float *dst, int width, int height,
                                    unsigned short white=1023,
                                    bool doBlackLevel = true, unsigned short black = 63);

        static void float32ToUint16(float *src, unsigned short *dst, int width, int height,
                                    unsigned short white=1023,
                                    bool doBlackLevel = true, unsigned short black = 63);
    };
}

#endif //  HUMANEFFECT_UTILS_FORMATTRANSFORM_H
