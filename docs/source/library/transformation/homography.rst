:mod:`transformation.homography` --- Homographies
=================================================

The :mod:`transformation.homography` module contains functions for computing a planar homography between two images and assessing reprojective error.  We do not generally suggest using these methods unless the user is confident that the images are planar.  In testing with planetary data, systematic error occurs due to topography even at long viewing distances.

.. versionadded:: 0.1.0

.. automodule:: autocnet.transformation.homography
   :synopsis: Planar Transformation Matrices and Error Computation
   :members:
