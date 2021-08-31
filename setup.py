from pathlib import Path
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
from setuptools import setup, Extension

c_ext = Extension('cext_acv', sources=['acv_explainers/cext_acv/_cext.cc'])
cy_ext = Extension('cyext_acv', ['acv_explainers/cyext_acv/cyext_acv.pyx'], extra_compile_args=['-fopenmp'],
                   extra_link_args=['-fopenmp'])
cy_extnopa = Extension('cyext_acv_nopa', ['acv_explainers/cyext_acv/cyext_acv_nopa.pyx'],
                       extra_compile_args=['-fopenmp'],
                       extra_link_args=['-fopenmp'])

cy_extcache = Extension('cyext_acv_cache', ['acv_explainers/cyext_acv/cyext_acv_cache.pyx'],
                        extra_compile_args=['-fopenmp'],
                        extra_link_args=['-fopenmp'])

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='acv-exp',
      author='Salim I. Amoukou',
      author_email='salim.ibrahim-amoukou@universite-paris-saclay.fr',
      version='1.0.1',
      description='ACV is a library that provides robust and accurate explanations for machine learning models',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/salimamoukou/acv00',
      include_dirs=[numpy.get_include()],
      cmdclass={'build_ext': build_ext},
      ext_modules=cythonize([c_ext, cy_ext, cy_extnopa, cy_extcache]),
      setup_requires=['numpy'],
      install_requires=['numpy', 'scipy', 'scikit-learn', 'matplotlib', 'pandas', 'tqdm', 'ipython', 'seaborn'],
      extras_require={'test': ['xgboost', 'lightgbm', 'catboost', 'pyspark', 'shap', 'rpy2 == 2.9.4']},
      packages=['acv_explainers', 'experiments'],
      license='MIT',
      zip_safe=False
      )
