"""Package installer."""
import os
from setuptools import setup
from setuptools import find_packages

LONG_DESCRIPTION = ''
if os.path.exists('README.md'):
    with open('README.md') as fp:
        LONG_DESCRIPTION = fp.read()

REQUIREMENTS = []
if os.path.exists('requirements.txt'):
    with open('requirements.txt') as fp:
        REQUIREMENTS = [line.strip() for line in fp]

setup(
    name='brontes',
    version='0.0.2',
    description=(
        'Brontes is your helping cyclops for pytorch models training.'
    ),
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    author='Joris Cadow, Antonio Foncubierta Rodriguez, Matteo Manica',
    author_email=(
        'joriscadow@gmail.com, '
        'antonio.foncubierta@gmail.com, '
        'drugilsberg@gmail.com'
    ),
    url='https://github.com/PhosphorylatedRabbits/brontes',
    license='MIT',
    install_requires=REQUIREMENTS,
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Topic :: Software Development :: Libraries :: Python Modules'
    ],
    packages=find_packages(),
)
