from setuptools import setup, find_packages

setup(
    name='oyLabCode',
    version='0.1.0',
    packages=find_packages(include=['oyLabCode', 'oyLabCode.*']),
    install_requires=[
        'PyYAML'
    ]
)
