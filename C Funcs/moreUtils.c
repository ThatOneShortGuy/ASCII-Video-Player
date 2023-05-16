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
} pixel;

typedef struct {
    unsigned char y;
    unsigned char cb;
    unsigned char cr;
} YCbCr;

void to_YCbCr(pixel *inpix, YCbCr *outpix) {
    outpix->y = (unsigned char) (0.299 * inpix->r + 0.587 * inpix->g + 0.114 * inpix->b);
    outpix->cb = (unsigned char) (128 - 0.168736 * inpix->r - 0.331264 * inpix->g + 0.5 * inpix->b);
    outpix->cr = (unsigned char) (128 + 0.5 * inpix->r - 0.418688 * inpix->g - 0.081312 * inpix->b);
}

void to_RGB(YCbCr *inpix, pixel *outpix) {
    outpix->r = (unsigned char) (inpix->y + 1.402 * (inpix->cr - 128));
    outpix->g = (unsigned char) (inpix->y - 0.344136 * (inpix->cb - 128) - 0.714136 * (inpix->cr - 128));
    outpix->b = (unsigned char) (inpix->y + 1.772 * (inpix->cb - 128));
}

int compare_YCbCr_values(YCbCr *pix1, YCbCr *pix2, int threshold) {
    return (abs(pix1->cb - pix2->cb) + abs(pix1->cr - pix2->cr) > threshold || abs(pix1->y - pix2->y) > threshold*2) && pix1->y > threshold;
}

#pragma auto_inline(on)
int get_pixel(pixel *pix, PyObject *img, int col, int row) {
    pix->b = *(unsigned char *) PyArray_GETPTR3(img, col, row, 0);
    pix->g = *(unsigned char *) PyArray_GETPTR3(img, col, row, 1);
    pix->r = *(unsigned char *) PyArray_GETPTR3(img, col, row, 2);
    return 0;
}

int row_continual(PyObject *img, int col, int row, int w, int threshold_of_change){
    pixel pix1;
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

void compute_row(pixel *output, int i, int wfreq, int h, int w, int freq, PyObject *img) {
    // printf("i: %d, wfreq: %d, h: %d, w: %d, freq: %d\n", i, wfreq, h, w, freq);

    int j, k;
    int r, g, b;
    // printf("i: %d, wfreq: %d, h: %d, w: %d, freq: %d\n", i, wfreq, h, w, freq);
    for (j = 0; j < wfreq; j++) {// for j in range(ceil(w/freq)):
        // Take the average of the next `freq` pixels
        // printf("i: %d, j: %d\n", i, j);
        r = 0;g = 0;b = 0;
        for (k = 0; k < freq; k++) {
            if (j * freq + k >= w) break;
            r += *(unsigned char *) PyArray_GETPTR3(img, i, j * freq + k, 0);
            g += *(unsigned char *) PyArray_GETPTR3(img, i, j * freq + k, 1);
            b += *(unsigned char *) PyArray_GETPTR3(img, i, j * freq + k, 2);
            // printf("r: %d, g: %d, b: %d, k: %d\n", r, g, b, k);
        }
        r /= k;
        g /= k;
        b /= k;
        // printf("i: %d, j: %d | r: %d, g: %d, b: %d\n\n",i, j, r, g, b);

        // Add the average to the output list
        output[i * wfreq + j].r = r;
        output[i * wfreq + j].g = g;
        output[i * wfreq + j].b = b;
    }
}

pixel *get_color_samples(PyObject *img, int freq) {
    int h = (int) PyArray_DIM(img, 0);
    int w = (int) PyArray_DIM(img, 1);

    int i;
    int wfreq = (int) ceil((double) w / (double) freq);

    pixel *output = (pixel *) malloc(h * wfreq * sizeof(pixel));

    for (i = 0; i < h; i++) { // for i in range(h):
        compute_row(output, i, wfreq, h, w, freq, img);
    }

    // printf("Finished\n");
    // for (i = 0; i < h*wfreq; i++) printf("output[%d]: %d, %d, %d\n", i, output[i].r, output[i].g, output[i].b);
    return output;
}

static PyObject *cget_color_samples(PyObject *self, PyObject *args) {
    // arg1 is the numpy array as np.uint8 of dimensions (h, w, 3)
    // pyoutput is the output list that is already structured
    // pyfreq is the frequency of the color samples as int
    PyObject *arg1, *pyfreq;
    PyObject *img;
    if (!PyArg_ParseTuple(args, "OO", &arg1, &pyfreq)) {
        printf("Error parsing args");
        return NULL;
    }
    img = PyArray_FROM_OTF(arg1, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    // Py_DECREF(arg1);
    if (img == NULL) return NULL;

    int freq = PyLong_AsLong(pyfreq);
    // Py_DECREF(pyfreq);

    int h = (int) PyArray_DIM(img, 0);
    int w = (int) PyArray_DIM(img, 1);

    int i, j, k;
    int wfreq = (int) ceil((double) w / (double) freq);

    PyObject *pyr, *pyg, *pyb;
    PyObject *output = PyList_New(h * wfreq);
    // printf("h * wfreq: %d\n", h * wfreq);
    PyObject *listPixel = PyList_New(3);
    int r, g, b;

    for (i = 0; i < h; i++) { // for i in range(h):
        for (j = 0; j < wfreq; j++) {// for j in range(ceil(w/freq)):
            // Take the average of the next `freq` pixels
            // printf("i: %d, j: %d\n", i, j);
            r = 0;
            g = 0;
            b = 0;
            for (k = 0; k < freq; k++) {
                if (j * freq + k >= w) break;
                r += *(unsigned char *) PyArray_GETPTR3(img, i, j * freq + k, 0);
                g += *(unsigned char *) PyArray_GETPTR3(img, i, j * freq + k, 1);
                b += *(unsigned char *) PyArray_GETPTR3(img, i, j * freq + k, 2);
                // printf("r: %d, g: %d, b: %d, k: %d\n", r, g, b, k);
            }
            r /= k;
            g /= k;
            b /= k;
            // printf("r: %d, g: %d, b: %d, k: %d\n\n", r, g, b, k);

            // Add the average to the output list
            pyr = PyLong_FromLong(r);
            pyg = PyLong_FromLong(g);
            pyb = PyLong_FromLong(b);
            PyList_SetItem(listPixel, 0, pyr);
            PyList_SetItem(listPixel, 1, pyg);
            PyList_SetItem(listPixel, 2, pyb);
            Py_DECREF(pyr);
            Py_DECREF(pyg);
            Py_DECREF(pyb);
            PyList_SetItem(output, i * wfreq + j, listPixel);
            listPixel = PyList_New(3);
        }
    }
    // Py_DECREF(img);
    // Py_DECREF(listPixel);
    // Py_DECREF(pyfreq);
    // Py_DECREF(arg1);

    return Py_BuildValue("O", output);

}