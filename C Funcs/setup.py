from setuptools import setup, Extension
import numpy as np
import os

if os.name == 'nt':
    compile_args = ['/O2', '/fp:fast', '/arch:AVX2', '/GL']
else:
    compile_args = ['-O3', '-ffast-math', '-march=native', '-funroll-loops']

module = Extension('cimg2ascii',
                   sources = ['C Funcs/utils.c'],
                   extra_compile_args=compile_args)

setup(name = 'cimg2ascii',
      version = '1.1',
      description = 'This is a package for img2ascii including map_color',
      ext_modules=[module],
      include_dirs=[np.get_include()],
      install_requires=['numpy', 'opencv-python'])