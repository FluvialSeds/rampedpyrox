.. rampedpyrox documentation master file, created by
   sphinx-quickstart on Mon Aug 22 18:31:16 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the RampedPyrox documentation
========================================

``rampedpyrox`` is a Python package for analyzing the results from ramped-
temperature instruments such as Ramped PyrOx, RockEval, pyrolysis gc (pyGC), 
thermogravimitry (TGA), etc. ``rampedpyrox`` uses a first-order Distributed 
Activation Energy Model (DAEM) to inversely determine the activation energy
(Ea) distribution corresponding to a sample's thermogram and to deconvolve
the Ea distribution into a discrete number of Gaussian peaks. Additionally, 
for isotope-enabled instruments such as Ramped PyrOx and pyGC-IRMS, 
``rampedpyrox`` calculates the best-fit stable carbon and radiocarbon value
for each Gaussian Ea peak.

When analyzing data with ``rampedpyrox`` to be used in a peer-reviewed
journal, please cite this package as:

J.D. Hemingway. *rampedpyrox*: open-source tools for thermoanalytical data
analysis, 2016-, http://github.com/FluvialSeds/rampedpyrox [online; accessed
|date|]

Additionally, please cite the following peer-reviewed manuscript describing
the deveopment of the package:

J.D. Hemingway et al. **(in prep)** Ramped-temperature decomposition kinetics
of organic matter using an inverse reactive continuum model.

Package information
-------------------
:Authors:
	Jordon D. Hemingway (jhemingway@whoi.edu)

:Version:
	0.0.2

:License:
	GNU GPL v3 (or greater)

:url:
	http://github.com/FluvialSeds/rampedpyrox

Package features
----------------
``rampedpyrox`` contains the following features:

* Stores and plots thermogram data

* Performs first-order DAEM inverse model (other models coming with *v.0.0.3*)

  * Smoothes f(Ea) using Tikhonov Regularization

    * Automated or user-defined regularization value

* Deconvolves f(Ea) distribution into Gaussian Peaks

  * Automated or user-defined peak number selection

* Calculates isotope values for each f(Ea) Gaussian peak

  * Can automatically blank-correct inputted values using calculated NOSAMS RPO 
    blank carbon composition

  * Allows for isotope determination of combined peaks

  * Determines peak radiocarbon (Fm) values

  * Determines peak stable-carbon (:sup:`13`\ C) ratios

  * Accounts for the kinetic isotope effect (KIE) during heating

    * Allows for unique KIE compensation for each peak

* Calculates and stores model performance metrics and goodness of fit statistics 

* Determines isotope value uncertainty using Monte Carlo resampling

* Allows for forward-modeling of any arbitrary time-temperature history, *e.g.* to 
  determine the decomposition rates and isotope fractionation during geologic 
  organic carbon matruation.


Table of contents
=================

.. toctree::
   :maxdepth: 2

   walkthrough
   model_dev
   package_reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`




.. |date| date::

