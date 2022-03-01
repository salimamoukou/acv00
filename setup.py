from pathlib import Path
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import numpy
# from setuptools import setup, Extension # for pypi build
from distutils.core import setup, Extension  # python setup

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
      version='1.2.1',
      description='ACV is a library that provides robust and accurate explanations for machine learning models or data',
      long_description=long_description,
      long_description_content_type="text/markdown",
      url='https://github.com/salimamoukou/acv00',
      include_dirs=[numpy.get_include()],
      cmdclass={'build_ext': build_ext},
      ext_modules=cythonize([c_ext, cy_ext, cy_extnopa, cy_extcache]),
      setup_requires=['numpy<1.22'],
      install_requires=['numpy<1.22', 'scipy', 'scikit-learn', 'matplotlib', 'pandas', 'tqdm', 'ipython', 'seaborn',
                        'streamlit', 'skranger == 0.7.0'],
      extras_require={'test': ['xgboost', 'lightgbm', 'catboost', 'pyspark', 'shap', 'rpy2 == 2.9.4']},
      packages=['acv_explainers', 'experiments', 'acv_app', 'acv_app.colors'],
      license='MIT',
      zip_safe=False
      )
