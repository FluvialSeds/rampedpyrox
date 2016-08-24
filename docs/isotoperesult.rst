Isotope Result
==============

This

This module creates ``EnergyComplex`` objects that contain the inversion model results,
deconvolves them into individual Gaussian peaks (with n either inputted by the user or
calculated automatically), and plots resulting Ea distribution.

Examples
--------

Running a thermogram through the inverse model and deconvolving Ea distribution::

	#load modules
	import matplotlib.pyplot as plt

	phi,resid_err,rgh_err,omega = lt.calc_fE_inv(tg,omega='auto')
	ec = rp.EnergyComplex(eps,phi,nPeaks='auto',combine_last=None)
	ax = ec.plot()

Technical documentation
-----------------------

(insert math here)

References
~~~~~~~~~~

(insert references here)

Module Reference
----------------
.. automodule:: rampedpyrox.core.energycomplex
	:members:
