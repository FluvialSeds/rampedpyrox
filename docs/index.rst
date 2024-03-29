.. rampedpyrox documentation master file, created by
   sphinx-quickstart on Mon Aug 22 18:31:16 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the RampedPyrox documentation
========================================

``rampedpyrox`` is a Python package for analyzing experimental kinetic data and accompanying chemical/isotope compositional information. ``rampedpyrox`` is especially suited for comparing kinetic and isotope results from ramped-temperature instruments such as Ramped PyrOx, RockEval, pyrolysis gc (pyGC), thermogravimitry (TGA), etc. This package converts measured time-series data into rate/activation energy distributions using a selection of reactive continuum models, including the Distributed Activation Energy Model (DAEM) for non-isothermal data. Additionally, this package calculates the range of rate/activation energy values associated with each isotope "fraction" and performs necessary isotope corrections (blank, mass balance, kinetic fractionation).

Package Information
-------------------
:Authors:
  Jordon D. Hemingway (jordon.hemingway@erdw.ethz.ch)

:Version:
  1.0.4

:Release:
  29 May 2023

:License:
  GNU GPL v3 (or greater)

:url:
  http://github.com/FluvialSeds/rampedpyrox
  http://pypi.python.org/pypi/rampedpyrox

:doi:
  |doi|

Bug Reports
-----------
This software is still in active deveopment. Please report any bugs directly to me.

How to Cite
-----------
When analyzing data with ``rampedpyrox`` to be used in a peer-reviewed
journal, please cite this package as:

* J.D. Hemingway. *rampedpyrox*: open-source tools for thermoanalytical data analysis, 2016-, http://pypi.python.org/pypi/rampedpyrox [online; accessed |date|]

Additionally, please cite the following peer-reviewed manuscript describing the deveopment of the package and Ramped PyrOx data treatment:

* J.D. Hemingway et al. (2017) Technical note: An inverse model to realte organic carbon reactivity to isotope composition from serial oxidation. *Biogeosciences*, **22**, 5099-5114.

If using Ramped PyrOx data generated by the NOSAMS instrument, the following manuscript contains relevant information regarding blank carbon composition, isotope mass balance, and the magnitude of the kinetic isotope effect:

* J.D. Hemingway et al. (2017) Assessing the blank carbon contribution, isotope mass balance, and kinetic isotope fractionation of the ramped pyrolysis/oxidation instrument at NOSAMS. *Radiocarbon*, **59**, 179-193.

Package features
----------------
``rampedpyrox`` currently contains the following features relevant to non-isothermal kinetic analysis:

* Stores and plots thermogram data

* Performs first-order DAEM inverse model to estimate activation energy distributions, p(0,E)

  * Regularizes ("smoothes") p(0,E) using Tikhonov Regularization

    * Automated or user-defined regularization value

* Calculates subset of p(0,E) contained in each RPO collection fraction

  * Automatically blank-corrects inputted isotope values using any known blank carbon composition

  * Corrects measured :sup:`13`\ C/:sup:`12`\ C ratios for the kinetic isotope effect (KIE) during heating

* Calculates and stores model performance metrics and goodness of fit 
  statistics

* Generates plots of thermograms, p(0,E), and E vs. isotope values for each RPO fraction

* Allows for forward-modeling of any arbitrary time-temperature history, *e.g.* to determine the decomposition rates and isotope fractionation during geologic organic carbon matruation

Future Additions
~~~~~~~~~~~~~~~~
Future versions of ``rampedpyrox`` will aim to include:

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
* :ref:`search`


.. * :ref:`modindex`

.. |date| date::
.. |doi| image:: https://zenodo.org/badge/66090463.svg