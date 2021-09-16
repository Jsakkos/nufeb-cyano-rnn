from setuptools import setup, find_packages

setup(
   name='nufebtools',
   version='0.0.1',
   description='Utilities for handing NUFEB setup, runs, and data',
   author='Joseph E. Weaver',
   author_email='joe.e.weaver@gmail.com',
   package_dir={'': 'src'},
   packages=find_packages('src'),
   install_requires=[],
   tests_require=['pytest']
)
