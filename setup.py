from distutils.core import setup, Extension
from Cython.Distutils import build_ext
import numpy
# from setuptools.command.build_ext import build_ext as _build_ext

# to publish use:
# > python setup.py sdist bdist_wheel upload
# which depends on ~/.pypirc

# Extend the default build_ext class to bootstrap numpy installation
# that are needed to build C extensions.
# see https://stackoverflow.com/questions/19919905/how-to-bootstrap-numpy-installation-in-setup-py
# class build_ext(_build_ext):
#     def finalize_options(self):
#         _build_ext.finalize_options(self)
#         if isinstance(__builtins__, dict):
#             __builtins__["__NUMPY_SETUP__"] = False
#         else:
#             setattr(__builtins__, "__NUMPY_SETUP__", False)
#         import numpy
#         print("numpy.get_include()", numpy.get_include())
#         self.include_dirs.append(numpy.get_include())

module1 = Extension('cext_acv', sources=['cext_acv/_cext.cc'])
module2 = Extension('exp_co', ['acv_explainers/exp_co.pyx'], extra_compile_args=['-fopenmp'],
                    extra_link_args=['-fopenmp'])

setup(name='acv',
      author='Salim I.Amoukou',
      author_email='salim.ibrahim-amoukou@universite-paris-saclay.fr',
      version='1.0',
      description='ACV function optimized in C',
      include_dirs=[numpy.get_include()],
      cmdclass={'build_ext': build_ext},
      ext_modules=[module1, module2],
      # cmdclass={'build_ext': build_ext},
      # setup_requires=['numpy'],
      # install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'pandas', 'tqdm', 'ipython'],
      packages=['acv_explainers', 'experiments'])
