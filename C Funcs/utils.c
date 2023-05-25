#ifndef INCLUDES
#define INCLUDES

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>

#endif

#include "moreUtils.c"

#define COLOR_SIZE 20

static PyObject *cpredict_insert_color_size(PyObject *self, PyObject *args){
    PyObject *img, *pythreshold_of_change;
    if (!PyArg_ParseTuple(args, "OO", &img, &pythreshold_of_change)) {
        printf("Error parsing args\n");
        return NULL;
    }
    img = PyArray_FROM_OTF(img, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    int threshold_of_change = (int) PyLong_AsLong(pythreshold_of_change);

    int h = (int) PyArray_DIM(img, 0);
    int w = (int) PyArray_DIM(img, 1);

    pixel current_color;
    YCbCr prev_YCbCr = { 255, 255, 255 };
    YCbCr YCbCr_color;

    int length = 0;
    int change;
    // printf("Threshold of change: %zd\n", threshold_of_change);

    for (int col=0; col < h; ++col) {
        for (int row=0; row < w; ++row) {
            get_pixel(&current_color, img, col, row);

            // Convert to YCbCr
            to_YCbCr(&current_color, &YCbCr_color);

            // if (YCbCr_color.y < threshold_of_change && row_continual(img, col, row, w, threshold_of_change)) break;

            if (compare_YCbCr_values(&YCbCr_color, &prev_YCbCr, threshold_of_change)) {
                if ((change=check_in_ansi_range(&YCbCr_color, threshold_of_change))==-1) {
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
        length+=3;
    }
    // printf("Length: %d\n", length);
    Py_DECREF(img);
    return PyLong_FromLong(length);

}

static PyObject *cinsert_color(PyObject *self, PyObject *args){
    PyObject *s, *img, *pythreshold_of_change;
    if (!PyArg_ParseTuple(args, "OOO", &s, &img, &pythreshold_of_change)) {
        printf("Error parsing args\n");
        return NULL;
    }
    
    Py_ssize_t string_len = PyUnicode_GetLength(s);
    img = PyArray_FROM_OTF(img, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    int threshold_of_change = (int) PyLong_AsLong(pythreshold_of_change);
    wchar_t *string = PyUnicode_AsWideCharString(s, &string_len);

    int h = (int) PyArray_DIM(img, 0);
    int w = (int) PyArray_DIM(img, 1);

    pixel current_color;
    YCbCr prev_YCbCr = { 255, 255, 255 };
    YCbCr YCbCr_color;
    int length = 0;
    wchar_t char_string;
    wchar_t *new_string = malloc(sizeof(wchar_t) * (string_len * 19 + 1));
    int change, len;
    
    for (int col=0; col < h; ++col) {
        for (int row=0; row < w; ++row) {
            get_pixel(&current_color, img, col, row);
            char_string = string[col*(w+1) + row];

            // Convert to YCbCr
            to_YCbCr(&current_color, &YCbCr_color);

            // If the color of the pixel and the rest of the row is too dark, just make a new line.
            // if (YCbCr_color.y < threshold_of_change && row_continual(img, col, row, w, threshold_of_change)) break;
            
            // Only change the color if the sum of the color differences (Cb + Cr) is greater than the threshold_of_change
            if (compare_YCbCr_values(&YCbCr_color, &prev_YCbCr, threshold_of_change)) {
                if ((change=check_in_ansi_range(&YCbCr_color, threshold_of_change))==-1) {
                    len = swprintf(new_string + length, COLOR_SIZE*sizeof(wchar_t), L"\033[38;2;%d;%d;%dm", current_color.r, current_color.g, current_color.b);
                } else {
                    len = swprintf(new_string + length, COLOR_SIZE*sizeof(wchar_t), L"\033[%dm", change);
                }
                // printf("%dx%d len: %d\n", row, col, len);
                if (len == -1) {
                    printf("Error copying color string\n");
                    return NULL;
                }
            
                length += len;
                prev_YCbCr = YCbCr_color;
            }
            // printf("length: %d, rowxcolumn: %d\n", length, col*(w+1) + row);
            new_string[length] = char_string;
            length += 1;
        }
        length += swprintf(new_string + length, COLOR_SIZE*sizeof(wchar_t), L"\033[E", change);;
        // wprintf(L"%ls", new_string);
    }
    PyObject *ret = PyUnicode_FromWideChar(new_string, length);
    free(new_string);
    // printf("Length: %d\n", length);
    PyMem_Free(string);
    Py_DECREF(img);
    return ret;
}

static PyMethodDef UtilsMethods[] = {
    {"cpredict_insert_color_size", (PyCFunction)cpredict_insert_color_size, METH_VARARGS, "Predict the size of the resulting string"},
    {"cinsert_color", (PyCFunction)cinsert_color, METH_VARARGS, "Insert color to string from image"},
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
