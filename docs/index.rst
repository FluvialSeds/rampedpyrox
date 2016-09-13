.. rampedpyrox documentation master file, created by
   sphinx-quickstart on Mon Aug 22 18:31:16 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the RampedPyrox documentation
========================================

``rampedpyrox`` is a Python package for analyzing experimental kinetic data 
and accompanying chemical/isotope compositional information. ``rampedpyrox`` 
is especially suited for deconvolution of results from ramped-temperature 
instruments such as Ramped PyrOx, RockEval, pyrolysis gc (pyGC), 
thermogravimitry (TGA), etc. This package deconvolves time-series kinetic data 
into rate/activation energy components using a selection of reactive continuum 
models, including the Distributed Activation Energy Model (DAEM) for 
non-isothermal data. Additionally, this package calculates the chemical/
isotope composition associated with each component, accounting for any kinetic 
fractionation effects, and uses a bootstrap method for constraining 
uncertainty.

Package Information
-------------------
:Authors:
  Jordon D. Hemingway (jhemingway@whoi.edu)

:Version:
  0.0.2

:Release:
  12 September 2016

:License:
  GNU GPL v3 (or greater)

:url:
  http://github.com/FluvialSeds/rampedpyrox

Bug Reports
-----------
This software is still in active deveopment. Please report any bugs directly to me.

How to Cite
-----------
When analyzing data with ``rampedpyrox`` to be used in a peer-reviewed
journal, please cite this package as:

* J.D. Hemingway. *rampedpyrox*: open-source tools for thermoanalytical data
  analysis, 2016-, http://github.com/FluvialSeds/rampedpyrox [online; accessed
  |date|]

Additionally, please cite the following peer-reviewed manuscripts describing
the deveopment of the package and Ramped PyrOx data treatment:

* J.D. Hemingway et al. **(in prep)** Assessing the blank carbon contribution, 
  isotope mass balance, and kinetic isotope fractionation of the ramped 
  pyrolysis/oxidation instrument at NOSAMS.

* J.D. Hemingway et al. **(in prep)** Ramped-temperature decomposition kinetics
  of organic matter using an inverse reactive continuum model.

**Access to rampedpyrox is currently private until the peer-reviewed manuscripts accompanything this package have been published.** Please `contact me directly <jhemingway@whoi.edu>`_ for access.

Package features
----------------
``rampedpyrox`` contains the following features:

* Stores and plots thermogram data

* Performs first-order DAEM inverse model

  * Smoothes f(Ea) using Tikhonov Regularization

    * Automated or user-defined regularization value

* Deconvolves f(Ea) distribution into Gaussian Peaks

  * Automated or user-defined peak number selection

* Calculates isotope values for each f(Ea) Gaussian peak

  * Can automatically blank-correct inputted values using calculated NOSAMS 
    RPO blank carbon composition

  * Allows for isotope determination of combined peaks

  * Determines peak radiocarbon (Fm) values

  * Determines peak stable-carbon (:sup:`13`\ C) ratios

  * Accounts for the kinetic isotope effect (KIE) during heating

    * Allows for unique KIE compensation for each peak

* Calculates and stores model performance metrics and goodness of fit 
  statistics 

* Determines isotope value uncertainty using Monte Carlo resampling

* Allows for forward-modeling of any arbitrary time-temperature history, 
  *e.g.* to determine the decomposition rates and isotope fractionation 
  during geologic organic carbon matruation

Future Additions
~~~~~~~~~~~~~~~~
Future versions of ``rampedpyrox`` will aim to include:

* Choice of non-Gaussian fitted peak shapes

* Better support for isothermal experimental conditions

* Non-first-order kinetic models

License
-------
This product is licensed under the GNU GPL license, version 3 or greater.


Table of contents
=================

.. toctree::
   :maxdepth: 2

   walkthrough
   package_reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




.. |date| date::

