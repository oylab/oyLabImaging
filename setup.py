from setuptools import setup, find_packages

setup(
    name='oyLabImaging',
    version='0.1.0',
    description='data processing code for the Oyler-Yaniv lab @HMS',
    author='Alon Oyler-Yaniv',
    url='https://github.com/alonyan/oyLabImaging',
    packages=find_packages(include=['oyLabImaging', 'oyLabImaging.*']),
    install_requires=[
        'PyYAML'
    ]
)
