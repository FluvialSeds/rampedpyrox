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

Package information
-------------------
:Authors:
	Jordon D. Hemingway (jhemingway@whoi.edu)

:Version:
	0.1.1 (Pre-release 27 August 2016)

:License:
	GNU GPL v3 (or greater)

:url:
	http://github.com/FluvialSeds/rampedpyrox

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

  * Allows for isotope determination of combined peaks

  * Determines peak radiocarbon (Fm) values

  * Determines peak stable-carbon (:sup:`13`\ C) values

  * Accounts for the kinetic isotope effect (KIE) during heating

    * Allows for unique KIE compensation for each peak

* Calculates and stores model performance metrics

* Determines peak shape and isotope value uncertainty using Monte Carlo resampling


Table of contents
=================

.. toctree::
   :maxdepth: 2

   package_reference


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

