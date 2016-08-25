.. rampedpyrox documentation master file, created by
   sphinx-quickstart on Mon Aug 22 18:31:16 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the RampedPyrox documentation
========================================

``rampedpyrox`` is a Python package for analyzing the results from ramped-
temperature instruments such as RampedPyrox, RockEval, pyrolysis gc (pyGC), 
thermogravimitry (TGA), etc. Rampedpyrox deconvolves Gaussian activiation energy 
peaks within a given sample using a Distributed Activation Energy Model (DAEM) and 
calculates the corresponding stable-carbon and radiocarbon isotope values for each peak.

:Authors:
	Jordon D. Hemingway <jordonhemingway@gmail.com>

:Version:
	0.1 (as of 23 August 2016)

:License:
	MIT License

When using ``rampedpyrox``, please cite the following peer-reviewed publications:

(insert papers here once published)

Minimal examples
----------------
The following examples should guide you through setting up a thermogram,
calculating activation energy (Ea) distributions, deconvolving into Gaussians,
and calculating isotope values.

Importing data into a thermogram object and plotting::

	#load modules
	import rampedpyrox as rp
	import matplotlib.pyplot as plt

	data = '/path_to_folder_containing_data/data.csv'
	nT = 250 #number of timepoints
	rd = rp.RealData(data,nT=nT)
	ax = rd.plot(xaxis='time')

Note that ``data`` can also be inputted as a ``pandas.Dataframe`` and must contain
'date_time', 'CO2_scaled', and 'temp' columns.

Calculating the Laplace Transform object and plotting the L-curve::

	#load modules
	import numpy as np

	eps = np.arange(50,350) #Ea range to calculate over
	logk0 = 10 #pre-exponential (Arrhenius) factor
	lt = rp.LaplaceTransform(rd.t,rd.Tau,eps,logk0)
	omega,ax = lt.plot_L_curve(rd)

Running a thermogram through the inverse model and deconvolving Ea distribution::

	phi,resid_err,rgh_err,omega = lt.calc_fE_inv(tg,omega='auto')
	ec = rp.EnergyComplex(eps,phi,nPeaks='auto',combine_last=None)
	ax = ec.plot()

Forward-modeling the resulting Ea distribution::
	
	#store in ModeledData object
	md = lt.calc_TG_fwd(ec)
	md.plot()

Calculating the isotope composition of each peak::

	#save string pointing to isotope data
	iso_data = '/path_to_folder_containing_data/isotope_data.csv'

	#create IsotopeResult object using data and ModeledResult object md
	ir = rp.IsotopeResult(iso_data,md)

	#view results
	print(ir.Fm_peak)
	print(ir.d13C_peak)

Table of contents
=================

.. toctree::
   :maxdepth: 2

   thermogram
   laplacetransform
   energycomplex
   isotoperesult

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

