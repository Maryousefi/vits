from distutils.core import setup
from Cython.Build import cythonize
import numpy
from setuptools.extension import Extension

extensions = [
    Extension("monotonic_align.core", ["core.pyx"], include_dirs=[numpy.get_include()])
]

setup(
    name='monotonic_align',
    ext_modules=cythonize(extensions),
)
