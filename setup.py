import os
from setuptools import setup, find_packages
import autocnet
from autocnet.examples import available
#Grab the README.md for the long description
with open('README.rst', 'r') as f:
    long_description = f.read()

def setup_package():
    examples = set()
    for i in available():
        if not os.path.isdir('autocnet/examples/' + i):
            if '.' in i:
                glob_name = 'examples/*.' + i.split('.')[-1]
            else:
                glob_name = 'examples/' + i
        else:
            glob_name = 'examples/' + i + '/*'
        examples.add(glob_name)

    setup(
        name = "autocnet",
        version = '0.2.5',
        author = "Jay Laura",
        author_email = "jlaura@usgs.gov",
        description = ("I/O API to support planetary data formats."),
        long_description = long_description,
        license = "Public Domain",
        keywords = "Multi-image correspondence detection",
        url = "http://packages.python.org/autocnet",
        packages=find_packages(),
        include_package_data=True,
        package_data={'autocnet' : list(examples)},
        zip_safe=False,
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Topic :: Utilities",
            "License :: Public Domain",
            'Programming Language :: Python :: 2.7',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.3',
            'Programming Language :: Python :: 3.4',
            'Programming Language :: Python :: 3.5',
        ],
    )

if __name__ == '__main__':
    setup_package()
