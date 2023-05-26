// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifndef INCLUDES
#define INCLUDES

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>

#endif

typedef struct {
    unsigned char r;
    unsigned char g;
    unsigned char b;
} Pixel;

typedef struct {
    unsigned char y;
    unsigned char cb;
    unsigned char cr;
} YCbCr;

void to_YCbCr(Pixel *inpix, YCbCr *outpix) {
    outpix->y = (unsigned char) (0.299 * inpix->r + 0.587 * inpix->g + 0.114 * inpix->b);
    outpix->cb = (unsigned char) (128 - 0.168736 * inpix->r - 0.331264 * inpix->g + 0.5 * inpix->b);
    outpix->cr = (unsigned char) (128 + 0.5 * inpix->r - 0.418688 * inpix->g - 0.081312 * inpix->b);
}

void to_RGB(YCbCr *inpix, Pixel *outpix) {
    outpix->r = (unsigned char) (inpix->y + 1.402 * (inpix->cr - 128));
    outpix->g = (unsigned char) (inpix->y - 0.344136 * (inpix->cb - 128) - 0.714136 * (inpix->cr - 128));
    outpix->b = (unsigned char) (inpix->y + 1.772 * (inpix->cb - 128));
}

int compare_YCbCr_values(YCbCr *pix1, YCbCr *pix2, int threshold) {
    return (abs(pix1->cb - pix2->cb) + abs(pix1->cr - pix2->cr) > threshold || abs(pix1->y - pix2->y) > threshold*2) && pix1->y > threshold;
}

int get_pixel(Pixel *pix, PyObject *img, int col, int row) {
    pix->b = *(unsigned char *) PyArray_GETPTR3(img, col, row, 0);
    pix->g = *(unsigned char *) PyArray_GETPTR3(img, col, row, 1);
    pix->r = *(unsigned char *) PyArray_GETPTR3(img, col, row, 2);
    return 0;
}

int predict_insert_color_size(int h, int w, PyObject *img, int threshold_of_change, int interlace_start, int interlace) {

    Pixel current_color;
    YCbCr prev_YCbCr = { 255, 255, 255 };
    YCbCr YCbCr_color;

    int length = 0;
    // printf("Threshold of change: %zd\n", threshold_of_change);
    if (interlace_start > 0) {
        length += 5;
    }

    for (int col=interlace_start; col < h; col+=interlace) {
        for (int row=0; row < w; ++row) {
            get_pixel(&current_color, img, col, row);

            // Convert to YCbCr
            to_YCbCr(&current_color, &YCbCr_color);

            // if (YCbCr_color.y < threshold_of_change && row_continual(img, col, row, w, threshold_of_change)) break;

            if (compare_YCbCr_values(&YCbCr_color, &prev_YCbCr, threshold_of_change)) {
                if (check_in_ansi_range(&YCbCr_color, threshold_of_change)==-1) {
                    length += 7;
                    length += 4*(current_color.r>=100) + 3*(current_color.r>=10 && current_color.r<100) + 2*(current_color.r<10); // Extra 1 for the semicolon
                    length += 4*(current_color.g>=100) + 3*(current_color.g>=10 && current_color.g<100) + 2*(current_color.g<10);
                    length += 5*(current_color.b>=100) + 4*(current_color.b>=10 && current_color.b<100) + 3*(current_color.b<10); // to account for "m_"
                } else {
                    length += 6;
                }
                prev_YCbCr = YCbCr_color;

            } else {
                length++;
            }
            // printf("Length: %d\n\n", length);
        }
        length += 3 + 1*(interlace > 1);
    }
    return length;
}

int get_freq(long freq, long min_freq, int h, int w, PyObject *pyframe, long max_chars, int interlace_start, int interlace) {
    while (freq > min_freq && predict_insert_color_size(h, w, pyframe, freq, interlace_start, interlace) < max_chars - 200) {
        freq -= 4;
    }
    freq = freq < min_freq ? min_freq : freq;
    while (predict_insert_color_size(h, w, pyframe, freq, interlace_start, interlace) > max_chars && freq < 255) {
        freq += 1;
    }
    return freq;
}

int row_continual(PyObject *img, int col, int row, int w, int threshold_of_change){
    Pixel pix1;
    YCbCr colors;
    while (row++ < w) {
        get_pixel(&pix1, img, col, row);
        to_YCbCr(&pix1, &colors);
        if (colors.y > threshold_of_change) return 0;
    }
    return 1;
}

int check_in_ansi_range(YCbCr *pix, int threshold) {
    YCbCr Black, DarkRed, DarkGreen, DarkYellow, DarkBlue, DarkMagenta, DarkCyan, DarkWhite, BrightBlack, BrightRed, BrightGreen, BrightYellow, BrightBlue, BrightMagenta, BrightCyan, White;
    Black.y = 0; Black.cb = 128; Black.cr = 128;
    DarkRed.y = 71; DarkRed.cb = 105; DarkRed.cr = 218;
    DarkGreen.y = 102; DarkGreen.cb = 78; DarkGreen.cr = 69;
    DarkYellow.y = 149; DarkYellow.cb = 44; DarkYellow.cr = 159;
    DarkBlue.y = 57; DarkBlue.cb = 219; DarkBlue.cr = 87;
    DarkMagenta.y = 71; DarkMagenta.cb = 173; DarkMagenta.cr = 174;
    DarkCyan.y = 131; DarkCyan.cb = 179; DarkCyan.cr = 76;
    DarkWhite.y = 204; DarkWhite.cb = 128; DarkWhite.cr = 128;
    BrightBlack.y = 118; BrightBlack.cb = 128; BrightBlack.cr = 128;
    BrightRed.y = 121; BrightRed.cb = 108; BrightRed.cr = 206;
    BrightGreen.y = 124; BrightGreen.cb = 65; BrightGreen.cr = 55;
    BrightYellow.y = 235; BrightYellow.cb = 89; BrightYellow.cr = 138;
    BrightBlue.y = 117; BrightBlue.cb = 206; BrightBlue.cr = 87;
    BrightMagenta.y = 72; BrightMagenta.cb = 177; BrightMagenta.cr = 205;
    BrightCyan.y = 179; BrightCyan.cb = 148; BrightCyan.cr = 70;
    White.y = 242; White.cb = 128; White.cr = 128;
    if (!compare_YCbCr_values(pix, &Black,         threshold/2)) return 30;
    if (!compare_YCbCr_values(pix, &DarkRed,       threshold/2)) return 31;
    if (!compare_YCbCr_values(pix, &DarkGreen,     threshold/2)) return 32;
    if (!compare_YCbCr_values(pix, &DarkYellow,    threshold/2)) return 33;
    if (!compare_YCbCr_values(pix, &DarkBlue,      threshold/2)) return 34;
    if (!compare_YCbCr_values(pix, &DarkMagenta,   threshold/2)) return 35;
    if (!compare_YCbCr_values(pix, &DarkCyan,      threshold/2)) return 36;
    if (!compare_YCbCr_values(pix, &DarkWhite,     threshold/4)) return 37;
    if (!compare_YCbCr_values(pix, &BrightBlack,   threshold/2)) return 90;
    if (!compare_YCbCr_values(pix, &BrightRed,     threshold/2)) return 91;
    if (!compare_YCbCr_values(pix, &BrightGreen,   threshold/2)) return 92;
    if (!compare_YCbCr_values(pix, &BrightYellow,  threshold/2)) return 93;
    if (!compare_YCbCr_values(pix, &BrightBlue,    threshold/2)) return 94;
    if (!compare_YCbCr_values(pix, &BrightMagenta, threshold/2)) return 95;
    if (!compare_YCbCr_values(pix, &BrightCyan,    threshold/2)) return 96;
    if (!compare_YCbCr_values(pix, &White,         threshold/4)) return 97;
    return -1;
}