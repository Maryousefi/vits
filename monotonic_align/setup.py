from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy
import os

# Absolute path to this directory
this_dir = os.path.abspath(os.path.dirname(__file__))
core_path = os.path.join(this_dir, "core.pyx")

extensions = [
    Extension(
        "monotonic_align.core",
        [core_path],
        include_dirs=[numpy.get_include()],
    )
]

setup(
    name="monotonic_align",
    ext_modules=cythonize(extensions, language_level="3"),
)
