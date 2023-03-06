# Usage:
# pip install -e .
from setuptools import setup, find_packages

setup(
    name='hotline',
    version='0.0.0',
    url='https://github.com/danielsnider/hotline',
    author='Daniel Snider',
    author_email='danielsnider12@gmail.com',
    description='Hotline',
    packages=find_packages(),
    install_requires=[
        'perfetto==0.5.0',
        'humanize>=4.4.0',
        'numpy>=1.21.1',
        'tabulate>=0.8.10',
        'torch>=1.12.0',
        'torchvision>=0.13.0',
        'pandas>=1.3.1',
        'torchinfo>=1.7.0',
        'orjson>=3.8.0',
        # Dev (always require)
        'ipython>=8.0.1',
        'pytest>=7.0.1',
    ],
    extras_require={
        'dev': [
        ]
    }
)
