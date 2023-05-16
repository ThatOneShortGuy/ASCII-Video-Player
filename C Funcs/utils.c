#ifndef INCLUDES
#define INCLUDES

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>

#endif

#include "moreUtils.c"

#define COLOR_SIZE 20
#define THREADS

static PyObject *cmap_color(PyObject *self, PyObject *args) {
    int THRESHOLD = 16;
    PyObject *pystring, *img, *pyline_len, *pyfreq, *pyOptimizations;
    if (!PyArg_ParseTuple(args, "OOOOO", &pystring, &img, &pyline_len, &pyfreq, &pyOptimizations)) {
        printf("Error parsing args\n");
        return NULL;
    }
    Py_ssize_t string_len = PyUnicode_GetLength(pystring);
    size_t line_len = PyLong_AsLong(pyline_len);
    wchar_t *string = PyUnicode_AsWideCharString(pystring, &string_len);
    // return PyUnicode_FromWideChar(string, string_len);
    // wprintf(L"string:\n%ls\n", string);
    // printf("len of string in bytes: %zd\n", wcslen(string)*sizeof(wchar_t));
    int freq = PyLong_AsLong(pyfreq);
    img = PyArray_FROM_OTF(img, NPY_UINT8, NPY_ARRAY_IN_ARRAY);
    pixel *img_colors = get_color_samples(img, freq);
    // for (int i = 0; i < 6;i++) printf("img_colors[%d]: %d, %d, %d\n", i, img_colors[i].r, img_colors[i].g, img_colors[i].b);

    size_t n1 = string_len / line_len;
    size_t n2 = line_len;
    int optimizations = PyLong_AsLong(pyOptimizations);
    // printf("n1: %zd, n2: %zd\n", n1, n2);
    // printf("n1 * n2 = %zd\n", n1 * n2);
    // printf("n1 * ceil(n2 / freq) = %zd\n", n1 * (n2 / freq + 1 ));
    // printf("COLOR_SIZE*(n2/freq)*n1 = %zd\n", COLOR_SIZE*(n2/freq+1)*n1);
    size_t size = sizeof(wchar_t) * (n2 + (COLOR_SIZE)*(n2/freq+1)) * n1;
    wchar_t *new_string = malloc(size+1);
    // memset(new_string, 0, size+1);
    // printf("Allocated new string successfully, size: %zd\n", size/sizeof(wchar_t));
    int j = 0;
    pixel current_color;
    pixel prev_color = { 255 };
    int current_pos = 0;
    // printf("string_len: %zd, line_len: %zd, freq: %d\n", string_len, line_len, freq);
    // printf("[");
    int copied;
    for (int i = 0; i < string_len; i++) {
        if ((i % line_len) % freq || string[i] == L'\n' || (current_pos > 15800 && optimizations)) {
            new_string[current_pos] = string[i];
            current_pos++;
            // printf("i: %d\n%c [%d]\n", i, string[i], string[i]);
            // printf("%d, ", string[i]);
            continue;
        }
        if (current_pos > 14400 && optimizations) {
            THRESHOLD = 50;
        } else if (current_pos > 15300 && optimizations) {
            THRESHOLD = 100;
        }
        current_color.b = img_colors[j].r;
        current_color.g = img_colors[j].g;
        current_color.r = img_colors[j].b;
        if (abs(current_color.r - prev_color.r) < THRESHOLD &&
            abs(current_color.g - prev_color.g) < THRESHOLD &&
            abs(current_color.b - prev_color.b) < THRESHOLD) {
            new_string[current_pos] = string[i];
            current_pos++;
            j++;
            continue;
        }
        // printf("r: %d, g: %d, b: %d\n", r, g, b);
        copied = swprintf(new_string + current_pos, COLOR_SIZE*2, L"\033[38;2;%d;%d;%dm%c", current_color.r, current_color.g, current_color.b, string[i]);
        // printf("[");
        // for (int k = 0; k < 620; k++) printf("%d, ", new_string[k]);
        // printf("] %c(%d)\n", string[i], string[i]);
        if (copied == -1) {
            printf("Error copying color string Part 2\n");
            return NULL;
        }
        current_pos += copied;
        prev_color = current_color;
        // printf("current_pos: %d\n", current_pos);
        j++;
    }
    // printf("]\n");
    // printf("[");
    // for (int i = 0; i < current_pos; i++) {
    //     printf("%d, ", new_string[i]);
    // }
    // printf("]\n");
    // printf("\033[0mSupposed size: %d\n", current_pos);
    // printf("Actual size: %zd\n", wcslen(new_string));
    PyObject *ret = PyUnicode_FromWideChar(new_string, current_pos);
    free(img_colors);
    // printf("check\n");
    free(string);
    free(new_string);
    Py_DECREF(img);
    // Py_DECREF(string_len);
    // Py_DECREF(ret);
    return ret;
}

static PyObject *cmap_color_old(PyObject *self, PyObject *args) {
    PyObject *pystring, *pycolors, *pyh, *pyfreq;
    if (!PyArg_ParseTuple(args, "OOOO", &pystring, &pycolors, &pyh, &pyfreq)) {
        printf("Error parsing args");
        return NULL;
    }
    Py_ssize_t string_len = PyUnicode_GetLength(pystring);
    wchar_t *string = malloc(sizeof(wchar_t) * string_len + 1);
    // printf("string_len: %lld\nAllocated space: %zd\n", , sizeof(wchar_t) * string_len);
    Py_ssize_t error = PyUnicode_AsWideChar(pystring, string, string_len);
    if (error == -1) {
        printf("Error converting string to wide char");
        return NULL;
    }
    // wprintf(L"string:\n%s\n", string);
    // size_t string_len = wcslen(string);
    unsigned char *colors = malloc(sizeof(int) * 3 * PyList_Size(pycolors));
    for (int i = 0; i < PyList_Size(pycolors); i++) {
        PyObject *pycolor = PyList_GetItem(pycolors, i);
        colors[i * 3] = (char) PyLong_AsLong(PyTuple_GetItem(pycolor, 0));
        colors[i * 3 + 1] = (char) PyLong_AsLong(PyTuple_GetItem(pycolor, 1));
        colors[i * 3 + 2] = (char) PyLong_AsLong(PyTuple_GetItem(pycolor, 2));
    }
    int h = PyLong_AsLong(pyh);
    int freq = PyLong_AsLong(pyfreq);
    // printf("num of colors: %lld\n", PyList_Size(pycolors));
    int size = (int) sizeof(wchar_t) * (COLOR_SIZE * PyList_Size(pycolors) + string_len + PyList_Size(pycolors));
    wchar_t *new_string = malloc(size);
    memset(new_string, 0, size);
    // printf("Allocated new string successfully, size: %d\n", size);
    // printf("string_len: %zd, h: %d, freq: %d\n", string_len, h, freq);
    int j = 0;
    int current_pos = 0;
    unsigned char r, g, b;
    wchar_t color_string[COLOR_SIZE] = { 0 };
    int other_num;
    for (int i = 0; i < string_len; i++) {
        if (j >= freq && j % freq == 0) {
            b = colors[3 * ((j-1) / freq)];
            g = colors[3 * ((j-1) / freq) + 1];
            r = colors[3 * ((j-1) / freq) + 2];
            int copied = swprintf(color_string, COLOR_SIZE, L"\033[38;2;");
            if (copied != 7) {
                printf("Error copying color string\n");
                return NULL;
            }
            other_num = swprintf(color_string + 7, 17, L"%d;%d;%dm%c", r, g, b, string[i]);
            if (other_num == -1) {
                printf("Error copying color string 2nd\n");
                printf("r: %d, g: %d, b: %d\n", r, g, b);
                for (int k = 0; k < 24; k++) printf("%c(%d) ", color_string[k], color_string[k]);
                wprintf(L"color_string:\n%s\n%d\n", color_string, string[i]);
                return NULL;
            }
            // color_string[7 + other_num] = L'\033';
            // color_string[7 + other_num + 1] = L'[';
            // color_string[7 + other_num + 2] = L'0';
            // color_string[7 + other_num + 3] = L'm';
            // wprintf(L"color_string:\n%s\n", color_string);
            
            for (int k = 0; k < COLOR_SIZE; k++) new_string[current_pos + k] = color_string[k];
            memset(color_string, 0, COLOR_SIZE*2);
            current_pos += COLOR_SIZE;
        }
        else {
            new_string[current_pos] = string[i];
            current_pos++;
        }
        j += (i % h != 0);
        
        // printf("i: %d, j: %d, current_pos: %d\n", i, j, current_pos);
    }
    free(colors);
    free(string);
    // printf("Should be: %d\n", size/2);
    return PyUnicode_FromWideChar(new_string, size/2);
}

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

    pixel prev_color = { 255, 255, 255};
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

            if (YCbCr_color.y < threshold_of_change && row_continual(img, col, row, w, threshold_of_change)) break;

            if (compare_YCbCr_values(&YCbCr_color, &prev_YCbCr, threshold_of_change)) {
                if ((change=check_in_ansi_range(&YCbCr_color, threshold_of_change))==-1) {
                    length += 7;
                    length += 4*(current_color.r>=100) + 3*(current_color.r>=10 && current_color.r<100) + 2*(current_color.r<10); // Extra 1 for the semicolon
                    length += 4*(current_color.g>=100) + 3*(current_color.g>=10 && current_color.g<100) + 2*(current_color.g<10);
                    length += 5*(current_color.b>=100) + 4*(current_color.b>=10 && current_color.b<100) + 3*(current_color.b<10); // to account for "m_"
                } else {
                    length += 6;
                }
                prev_color.r = current_color.r;
                prev_color.g = current_color.g;
                prev_color.b = current_color.b;
                prev_YCbCr = YCbCr_color;

            } else {
                length++;
            }
            // printf("Length: %d\n\n", length);
        }
        length++;
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

    pixel prev_color = { 255, 255, 255 };
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
            if (YCbCr_color.y < threshold_of_change && row_continual(img, col, row, w, threshold_of_change)) break;
            
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
                prev_color = current_color;
                prev_YCbCr = YCbCr_color;
            }
            // printf("length: %d, rowxcolumn: %d\n", length, col*(w+1) + row);
            new_string[length] = char_string;
            length += 1;
        }
        new_string[length] = L'\n';
        length += 1;
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
    {"cget_color_samples", (PyCFunction)cget_color_samples, METH_VARARGS, "Get color samples from image"},
    {"cmap_color", (PyCFunction)cmap_color, METH_VARARGS, "Map color to string"},
    {"cmap_color_old", (PyCFunction)cmap_color_old, METH_VARARGS, "Map color to string (Old version)"},
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
