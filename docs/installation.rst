Installation
============

We provide AutoCnet as a binary package via conda and for
installation via the standard setup.py script.

Via Conda
---------

1. [Download](https://www.continuum.io/downloads) and install the Python 3.x Miniconda installer.  Respond ``Yes`` when
   prompted to add conda to your BASH profile.
1. (Optional) We like to sequester applications in their own environments to avoid any dependency conflicts.  To do this:
  * ``conda create -n <your_environment_name> python=3 && source activate <your_environment_name>``
1. Bring up a command line and add three channels to your conda config (``~/condarc``):
  * ``conda config --add channels conda-forge``
  * ``conda condig --add channels jlaura``
  * ``conda config --add channels menpo``
1. Finally, install autocnet: ``conda install -c jlaura autocnet-dev``

Via setup.py
------------
This method assumes that you have the necessary dependencies already
installed. The installation of dependencies can be non-trivial because of GDAL.
We supply an ``environment.yml`` file that works with Anaconda Python's ``conda
env`` environment management tool.

Manual Development Environment
------------------------------
To manually install AutoCnet (for example in a development environment) we must install the necessary dependencies.

1. Create a virtual environment:  ``conda create -n autocnet python=3.5 && source activate autocnet``
2. As above, add conda-forge to the channel list.
3. To install the planetary I/O module and the OpenCV computer vision module: ``conda install -c jlaura plio opencv3``
4. To install the optional VLFeature module (for SIFT): ``conda install -c conda-forge vlfeat``
4a. To install the Cython wrapper to vlfeat: ``conda install -c menpo cyvlfeat``
5. To install PIL and PySAL: ``pip install pillow pysal``
6. To install additional conda packages: ``conda install scipy networkx numexpr dill cython pyyaml matplotlib``

This ensures that all dependencies needed to run AutoCnet are availble.  We also have development dependencies to
support automated testing, documentation builds, etc.

1. Install Nose and Sphinx: ``conda install nose sphinx``
2. Install coveralls: ``pip install coveralls``
3. Install the nbsphinx plugin: ``pip install nbshpinx``
4. Install Jupyter for notebook support: ``conda install jupyter``
