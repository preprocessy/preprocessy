.. preprocessy documentation master file, created by
   sphinx-quickstart on Fri Apr  2 00:27:40 2021.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Preprocessy's documentation!
=======================================

Preprocessy is a library that provides data preprocessing pipelines for machine
learning. It bundles all the common preprocessing steps that are performed on the
data to prepare it for machine learning models. It aims to do so in a manner that
is independent of the source and type of dataset. Hence, it provides a set of
functions that have been generalised to different types of data.

The pipelines themselves are composed of these functions and flexible so that
the users can customise them by adding their processing functions or removing
pipeline functions according to their needs. The pipelines thus provide an abstract
and high-level interface to the users.

API Reference
-------------

.. toctree::
   :maxdepth: 2

   api


Miscellaneous Pages
-------------------

.. toctree::
   :maxdepth: 2

   license
   contributing
   changes

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
