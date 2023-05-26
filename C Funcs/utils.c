#ifndef INCLUDES
#define INCLUDES

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>

#endif

#include "moreUtils.c"

#define COLOR_SIZE 20

static PyObject *py_predict_insert_color_size(PyObject *self, PyObject *args) {
    PyObject *img, *pythreshold_of_change, *pyinterlace_start, *pyinterlace;
    if (!PyArg_ParseTuple(args, "OOOO", &img, &pythreshold_of_change, &pyinterlace_start, &pyinterlace)) {
        printf("Error parsing args\n");
        return NULL;
    }
    img = PyArray_FROM_OTF(img, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    int threshold_of_change = (int) PyLong_AsLong(pythreshold_of_change);
    int interlace_start = (int) PyLong_AsLong(pyinterlace_start);
    int interlace = (int) PyLong_AsLong(pyinterlace);
    
    int h = (int) PyArray_DIM(img, 0);
    int w = (int) PyArray_DIM(img, 1);

    int length = predict_insert_color_size(h, w, img, threshold_of_change, interlace_start, interlace);
    
    // printf("Length: %d\n", length);
    Py_DECREF(img);
    return PyLong_FromLong(length);

}

static PyObject *py_insert_color(PyObject *self, PyObject *args) {
    PyObject *s, *img, *pythreshold_of_change, *pyinterlace_start, *pyinterlace;
    if (!PyArg_ParseTuple(args, "OOOOO", &s, &img, &pythreshold_of_change, &pyinterlace_start, &pyinterlace)) {
        printf("Error parsing args\n");
        return NULL;
    }
    
    Py_ssize_t string_len = PyUnicode_GetLength(s);
    img = PyArray_FROM_OTF(img, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    int threshold_of_change = (int) PyLong_AsLong(pythreshold_of_change);
    int interlace_start = (int) PyLong_AsLong(pyinterlace_start);
    int interlace = (int) PyLong_AsLong(pyinterlace);

    wchar_t *string = PyUnicode_AsWideCharString(s, &string_len);

    int h = (int) PyArray_DIM(img, 0);
    int w = (int) PyArray_DIM(img, 1);

    Pixel current_color;
    YCbCr prev_YCbCr = { 255, 255, 255 };
    YCbCr YCbCr_color;
    int length = 0;
    wchar_t char_string;
    wchar_t *new_string = malloc(sizeof(wchar_t) * (string_len * 19 + 1));
    int change, len;

    if (interlace_start > 0) {
        length += swprintf(new_string + length, 5, L"\033[%dE", interlace_start);
    }
    
    for (int col=interlace_start; col < h; col+=interlace) {
        for (int row=0; row < w; ++row) {
            get_pixel(&current_color, img, col, row);
            char_string = string[col*(w+1) + row];

            // Convert to YCbCr
            to_YCbCr(&current_color, &YCbCr_color);

            // If the color of the Pixel and the rest of the row is too dark, just make a new line.
            // if (YCbCr_color.y < threshold_of_change && row_continual(img, col, row, w, threshold_of_change)) break;
            
            // Only change the color if the sum of the color differences (Cb + Cr) is greater than the threshold_of_change
            if (compare_YCbCr_values(&YCbCr_color, &prev_YCbCr, threshold_of_change)) {
                if ((change=check_in_ansi_range(&YCbCr_color, threshold_of_change))==-1) {
                    len = swprintf(new_string + length, COLOR_SIZE*sizeof(wchar_t), L"\033[38;2;%d;%d;%dm", current_color.r, current_color.g, current_color.b);
                } else {
                    len = swprintf(new_string + length, 7, L"\033[%dm", change);
                }
            
                length += len;
                prev_YCbCr = YCbCr_color;
            }
            // printf("length: %d, rowxcolumn: %d\n", length, col*(w+1) + row);
            new_string[length] = char_string;
            length += 1;
        }
        length += (interlace == 1) ? swprintf(new_string + length, 4, L"\033[E") : swprintf(new_string + length, 6, L"\033[%dE", interlace);
        // wprintf(L"%ls", new_string);
    }
    PyObject *ret = PyUnicode_FromWideChar(new_string, length);
    free(new_string);
    // printf("Length: %d\n", length);
    PyMem_Free(string);
    Py_DECREF(img);
    return ret;
}

static PyObject *py_img2ascii(PyObject *self, PyObject *args) {
    PyObject *pyimg, *pyascii_map;
    if (!PyArg_ParseTuple(args, "OO", &pyimg, &pyascii_map)) {
        printf("Error parsing args in img2ascii\n");
        return NULL;
    }
    pyimg = PyArray_FROM_OTF(pyimg, NPY_UINT8, NPY_ARRAY_IN_ARRAY);

    wchar_t ascii_map[256];
    PyUnicode_AsWideChar(pyascii_map, ascii_map, 256);

    int h = (int) PyArray_DIM(pyimg, 0);
    int w = (int) PyArray_DIM(pyimg, 1);

    Pixel BGR_color;
    YCbCr YCbCr_color;

    int length = 0;
    wchar_t *string = malloc(sizeof(wchar_t) * (h * (w+1) + 1));

    for (int col=0;col<h;++col) {
        for (int row=0;row<w;++row) {
            get_pixel(&BGR_color, pyimg, col, row);
            to_YCbCr(&BGR_color, &YCbCr_color);
            string[length] = ascii_map[YCbCr_color.y];
            length++;
        }
        string[length] = '\n';
        length++;
    }
    PyObject *ret = PyUnicode_FromWideChar(string, length);
    free(string);
    Py_DECREF(pyimg);
    return ret;
}

static PyObject *py_get_freq(PyObject *self, PyObject *args) {
    PyObject *pyfreq, *pymin_freq, *pyframe, *pymax_chars, *pyinterlace_start, *pyinterlace;
    if (!PyArg_ParseTuple(args, "OOOOOO", &pyfreq, &pymin_freq, &pyframe, &pymax_chars, &pyinterlace_start, &pyinterlace)) {
        printf("Error parsing args in get_freq\n");
        return NULL;
    }
    long freq = PyLong_AsLong(pyfreq);
    long min_freq = PyLong_AsLong(pymin_freq);
    pyframe = PyArray_FROM_OTF(pyframe, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    long max_chars = PyLong_AsLong(pymax_chars);
    long interlace_start = PyLong_AsLong(pyinterlace_start);
    long interlace = PyLong_AsLong(pyinterlace);

    int h = (int) PyArray_DIM(pyframe, 0);
    int w = (int) PyArray_DIM(pyframe, 1);

    freq = get_freq(freq, min_freq, h, w, pyframe, max_chars, interlace_start, interlace);

    Py_DECREF(pyframe);
    return PyLong_FromLong(freq);
}

static PyMethodDef UtilsMethods[] = {
    {"cpredict_insert_color_size", (PyCFunction)py_predict_insert_color_size, METH_VARARGS, "Predict the size of the resulting string"},
    {"cinsert_color", (PyCFunction)py_insert_color, METH_VARARGS, "Insert color to string from image"},
    {"cimg2ascii", (PyCFunction)py_img2ascii, METH_VARARGS, "Convert image to ascii"},
    {"cget_freq", (PyCFunction)py_get_freq, METH_VARARGS, "Get the optimal frequency of the frame"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef utilsmodule = {
    PyModuleDef_HEAD_INIT,
    "cimg2ascii",
    "This is a package for img2ascii including map_color",
    -1,
    UtilsMethods
};

PyMODINIT_FUNC PyInit_cimg2ascii(void) {
    import_array();
    return PyModule_Create(&utilsmodule);
}
