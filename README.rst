===============================
AutoCNet
===============================

.. image:: https://badges.gitter.im/USGS-Astrogeology/autocnet.svg
   :alt: Join the chat at https://gitter.im/USGS-Astrogeology/autocnet
   :target: https://gitter.im/USGS-Astrogeology/autocnet?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge&utm_content=badge

.. image:: https://img.shields.io/pypi/v/autocnet.svg
        :target: https://pypi.python.org/pypi/autocnet

.. image:: https://travis-ci.org/USGS-Astrogeology/autocnet.svg?branch=master
    :target: https://travis-ci.org/USGS-Astrogeology/autocnet

.. image:: https://coveralls.io/repos/USGS-Astrogeology/autocnet/badge.svg?branch=master&service=github
    :target: https://coveralls.io/github/USGS-Astrogeology/autocnet?branch=master

.. image:: https://readthedocs.org/projects/autocnet/badge/?version=latest
    :target: http://autocnet.readthedocs.org/en/latest/
    :alt: Documentation Status

.. image:: https://badge.waffle.io/USGS-Astrogeology/autocnet.png?label=ready&title=Ready 
 :target: https://waffle.io/USGS-Astrogeology/autocnet 
 :alt: 'Stories in Ready'

Automated sparse control network generation to support photogrammetric control of planetary image data.

* Documentation: https://autocnet.readthedocs.org.

Installation Instructions
-------------------------
We suggest using Anaconda Python to install Autocnet within a virtual environment.  These steps will walk you through the process.

1. [Download](https://www.continuum.io/downloads) and install the Python 3.x Miniconda installer.  Respond ``Yes`` when
   prompted to add conda to your BASH profile.
1. (Optional) We like to sequester applications in their own environments to avoid any dependency conflicts.  To do this:
  * ``conda create -n <your_environment_name> python=3 && source activate <your_environment_name>``
1. Bring up a command line and add three channels to your conda config (``~/condarc``):
  * ``conda config --add channels conda-forge``
  * ``conda config --add channels jlaura``
  * ``conda config --add channels menpo``
1. Finally, install autocnet: ``conda install -c jlaura autocnet-dev``
