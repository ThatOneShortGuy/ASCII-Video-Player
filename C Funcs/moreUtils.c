// #define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <Python.h>
#include <stdio.h>
#include <stdlib.h>
#include <numpy/arrayobject.h>

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

    int h = PyArray_DIM(img, 0);
    int w = PyArray_DIM(img, 1);

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