from setuptools import setup

setup(
   name='sensei',
   description='Spectral Encoding for Sensor Independence',
   author='Alistair Francis and John Mrziglod',
   author_email='a.francis.16@ucl.ac.uk',
   packages=['sensei',
             'sensei/data'
             ],
   license='GNU General Public License v3 (GPLv3)',
   long_description=open('README.md').read()
)
