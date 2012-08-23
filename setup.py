from distutils.core import setup

# Keeping all Python code for package in lib directory
NAME = 'glm'
VERSION = '0.1'
AUTHOR = 'Alexander W Blocker'
AUTHOR_EMAIL = 'ablocker@gmail.com'
URL = 'http://www.awblocker.com'
DESCRIPTION = 'A lean, mean GLM-modeling machine in Python'

REQUIRES = ['numpy(>=1.6)','scipy(>=0.9)']

PACKAGE_DIR = {'': 'lib'}
PACKAGES = ['glm']
SCRIPTS = ['scripts/estimate_glm']

setup(name=NAME,
      url=URL,
      version=VERSION,
      description=DESCRIPTION,
      author=AUTHOR,
      author_email=AUTHOR_EMAIL,
      packages=PACKAGES,
      package_dir=PACKAGE_DIR,
      scripts=SCRIPTS,
      requires=REQUIRES
      )

