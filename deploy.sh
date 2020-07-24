travis-sphinx -v deploy -b dev
source deactivate
conda install conda-build anaconda-client
conda config --set anaconda_upload no
conda config --set channel_priority strict

travis-wait-improved --timeout 30m conda build --python $PYTHON_VERSION -c conda-forge -c usgs-astrogeology conda
