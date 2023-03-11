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

static PyMethodDef UtilsMethods[] = {
    {"cget_color_samples", (PyCFunction)cget_color_samples, METH_VARARGS, "Get color samples from image"},
    {"cmap_color", (PyCFunction)cmap_color, METH_VARARGS, "Map color to string"},
    {"cmap_color_old", (PyCFunction)cmap_color_old, METH_VARARGS, "Map color to string (Old version)"},
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
