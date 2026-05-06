from setuptools import setup, find_packages

setup(
    name='dist_fitter',
    version='0.1.0',
    packages=find_packages(),
    py_modules=['dist_fitter'],
    install_requires=[
        'numpy',
        'matplotlib',
        'scipy'
    ],
    description='A dynamic distribution fitting and plotting tool.',
    author='Huw Williams',
)
