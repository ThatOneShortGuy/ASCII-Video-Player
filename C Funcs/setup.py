from distutils.core import setup, Extension
import numpy as np

module = Extension('cimg2ascii', sources = ['C Funcs/utils.c'])

setup(name = 'cimg2ascii',
      version = '1.0',
      description = 'This is a package for img2ascii including map_color',
      ext_modules=[module],
      include_dirs=[np.get_include()])