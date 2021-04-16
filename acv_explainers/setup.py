# cython: linetrace=True


from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

from Cython.Compiler.Options import get_directive_defaults
directive_defaults = get_directive_defaults()

directive_defaults['linetrace'] = True
directive_defaults['binding'] = True

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[Extension('cyext_acv', ['cyext_acv.pyx'])],
      include_dirs=[numpy.get_include()],
      define_macros=[('CYTHON_TRACE_NOGIL', '1')]
)