travis-sphinx -v deploy -b dev
source deactivate
conda install conda-build anaconda-client
conda config --set anaconda_upload yes
conda build --token $CONDA_UPLOAD_TOKEN --python $PYTHON_VERSION -c conda-forge -c menpo -c usgs-astrogeology conda
