from setuptools import setup, Extension
import numpy as np

module = Extension('cimg2ascii',
                   sources = ['C Funcs/utils.c'],
                   extra_compile_args=['/O2'])

setup(name = 'cimg2ascii',
      version = '1.0',
      description = 'This is a package for img2ascii including map_color',
      ext_modules=[module],
      include_dirs=[np.get_include()],
      install_requires=['numpy', 'opencv-python', 'python-vlc'])