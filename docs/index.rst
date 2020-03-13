.. figure:: _static/images/matched.png
   :alt: Sample Matching
   :align: center

   A sample of the AutoCNet library being used for pseudo-dense matching on straight-forward Apollo 15 metric camera images.  Blue points are identified correspondences and yellow lines indicate that the points are linked (correspond with one another).

AutoCNet
========
AutoCNet is an open source library (API) for the pseudo-automated generation of sparse n-image
control networks.  AutoCNet is licensed in `the public domain`_.

AutoCNet supports the following:

- 32-bit CUDA based (GPU) keypoint extraction
- Support for images of arbitrary size using downsampling or tiling for feature extraction
- A reference implementation (the CandidateGraph) to demonstrate use of the API

Tutorials
---------
An example is worth 1000 pages of documentation.  Checkout our tutorials (jupyter notebooks).

.. toctree::
   :maxdepth: 1

   users/index

Issues and Version Control
--------------------------
This project is hosted on `Github`_. If you run into a problem, please open an `issue`_ on our `issue`_ tracker.

Documentation
-------------
.. toctree::
  :maxdepth: 4

  library/index

Developers
----------
.. toctree::
   :maxdepth: 2

   developers/index

References
----------
.. toctree::
   :maxdepth: 1

   references

License
-------
.. toctree::
   :maxdepth: 1

   license

.. _GitHub: http://github.com/USGS-Astrogeology/autocnet
.. _issue: https://github.com/USGS-Astrogeology/autocnet/issues?state=open
.. _the public domain: license.html
