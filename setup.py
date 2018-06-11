from setuptools import setup
from distutils.extension import Extension
import numpy

binaries=['bin/outlierml']

setup(
    name='outlierml',
    version='0.1',
    description='Outlier detection library',
    url='http://github.com/jsosa/outlierml',
    author='Jeison Sosa',
    author_email='sosa.jeison@gmail.com',
    license='MIT',
    packages=['outlierml'],
    zip_safe=False,
    scripts=binaries,
    )
