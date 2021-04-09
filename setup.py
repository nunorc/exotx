
from setuptools import setup, find_packages

with open('README.rst', 'r') as fh:
    long_description = fh.read()

setup(name='exotx',
      version='0.0.1a1',
      url='https://github.com/nunorc/exotx',
      author='Nuno Carvalho',
      author_email='narcarvalho@gmail.com',
      description='exoplanets transit explorer',
      long_description=long_description,
      long_description_content_type='text/x-rst',
      license='MIT',
      packages=find_packages(),
      install_requires=['lightkurve>=2.0.1','numpy','scipy','matplotlib','batman-package','emcee','astroquery','tqdm'])
