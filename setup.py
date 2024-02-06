from setuptools import setup

setup(
   name='help',
   version='1.0',
   author='Maurizio Giordano and Ilaria Granata and Lucia Maddalena',
   author_email='maurizio.giordano@icar.cnr.it',
   packages=['help', 'help.models', 'help.utility', 'help.visualization'],
   license='LICENSE.txt',
   description='The HELP Essential Genes framework',
   long_description=open('README.md').read(),
   install_requires=[
       "numpy",
       "tqdm",
       "typing",
       "pandas",
       "supervenn",
       "matplotlib",
       "ipywidgets",
       "scipy"
   ],
)

