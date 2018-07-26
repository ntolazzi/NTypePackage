from setuptools import setup, find_packages

setup(
    name='ntypecqed',
    version='0.1',
    packages=find_packages(exclude=('tests', 'docs', 'examples')),
    url='',
    license='MIT',
    author='Nicolas Tolazzi',
    author_email='nicolas.tolazzi@mpq.mpg.de',
    description='A package for simulating Cavity QED with N-type atoms',
    install_requires=['qutip', 'numpy']
)
