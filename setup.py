from setuptools import setup, find_packages

setup(
    name='oyLabCode',
    version='0.1.0',
    description='data processing code for the Oyler-Yaniv lab @HMS',
    author='Alon Oyler-Yaniv',
    url='https://github.com/alonyan/oyLabCode',
    packages=find_packages(include=['oyLabCode', 'oyLabCode.*']),
    install_requires=[
        'PyYAML'
    ]
)
