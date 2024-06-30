from setuptools import setup

setup(
   name='HELPpy',
   version='2.0',
   author='Maurizio Giordano and Ilaria Granata and Lucia Maddalena',
   author_email='maurizio.giordano@cnr.it',
   packages=['HELPpy', 'HELPpy.preprocess', 'HELPpy.utility', 'HELPpy.visualization'],
   #packages=['help', 'help.preprocess', 'help.models', 'help.utility', 'help.visualization'],
   license='LICENSE.txt',
   description='The Human gene Essentiality Labelling & Prediction (HELP) framework implementation in Python',
   long_description=open('README.md').read(),
   install_requires=[
       "numpy",
       "tqdm",
       "typing",
       "pandas",
       "supervenn",
       "matplotlib",
       "ipywidgets",
       "scikit-learn",
       "scipy"
   ],
)

