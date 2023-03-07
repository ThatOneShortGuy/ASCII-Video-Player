from distutils.core import setup, Extension

module = Extension('cimg2ascii', sources = ['C Funcs/utils.c'])

setup(name = 'cimg2ascii',
      version = '1.0',
      description = 'This is a package for img2ascii including map_color',
      ext_modules=[module])