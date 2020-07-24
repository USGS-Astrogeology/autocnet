travis-sphinx -v deploy -b dev
source deactivate
conda install conda-build anaconda-client
conda config --set anaconda_upload no
conda config --set channel_priority strict

pip install https://files.pythonhosted.org/packages/b1/66/0acc61f28ba6081da470120a503cc8855da8bab35d355ce037463486d362/travis_wait_improved-1.0.0-py3-none-any.whl

travis-wait-improved --timeout 30m conda build --python $PYTHON_VERSION -c conda-forge -c usgs-astrogeology conda
