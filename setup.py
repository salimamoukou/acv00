from distutils.core import setup, Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy

c_ext = Extension('cext_acv', sources=['acv_explainers/cext_acv/_cext.cc'])
cy_ext = Extension('cyext_acv', ['acv_explainers/cyext_acv/cyext_acv.pyx'], extra_compile_args=['-fopenmp'],
                    extra_link_args=['-fopenmp'])

cy_extnopa = Extension('cyext_acv_nopa', ['acv_explainers/cyext_acv/cyext_acv_nopa.pyx'], extra_compile_args=['-fopenmp'],
                    extra_link_args=['-fopenmp'])


cy_extcache = Extension('cyext_acv_cache', ['acv_explainers/cyext_acv/cyext_acv_cache.pyx'], extra_compile_args=['-fopenmp'],
                    extra_link_args=['-fopenmp'])

setup(name='acv',
      author='Salim I.Amoukou',
      author_email='salim.ibrahim-amoukou@universite-paris-saclay.fr',
      version='1.0',
      description='ACV function optimized in C',
      include_dirs=[numpy.get_include()],
      cmdclass={'build_ext': build_ext},
      ext_modules=cythonize([c_ext, cy_ext, cy_extnopa, cy_extcache]),
      # cmdclass={'build_ext': build_ext},
      # setup_requires=['numpy'],
      # install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'pandas', 'tqdm', 'ipython'],
      packages=['acv_explainers', 'experiments'],
      # language="c++"
      )
