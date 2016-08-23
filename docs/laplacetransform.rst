Laplace Transform
=================

This module creates ``LaplaceTransform`` objects to store the A matrix for calculating
inverse/forward model results, calculates, and plots the Tikhonov regularization 
L-curve.

Examples
--------

Calculating the Laplace Transform object and plotting the L-curve::

	#load modules
	import numpy as np

	eps = np.arange(50,350) #Ea range to calculate over
	logk0 = 10 #pre-exponential (Arrhenius) factor
	lt = rp.LaplaceTransform(rd.t,rd.Tau,eps,logk0)
	omega,axis = lt.plot_L_curve()

Technical documentation
-----------------------

(insert math here)

References
~~~~~~~~~~

Distributed activation energy model references:

1. R.L Braun and A.K. Burnham (1987) Analysis of chemical reaction kinetics
using a distribution of activation energies and simpler models. 
*Energy & Fuels*, **1**, 153-161.

2. B. Cramer et al. (1998) Modeling isotope fractionation during primary 
cracking of natural gas: A reaction kinetic approach. *Chemical 
Geology*, **149**, 235-250.

Inversion references:

1. D.C. Forney and D.H. Rothman (2012) Common structure in the 
heterogeneity of plant-matter decay. *Journal of the Royal Society 
Interface*, rsif.2012.0122.

2. D.C. Forney and D.H. Rothman (2012) Inverse method for calculating 
respiration rates from decay time series. *Biogeosciences*, **9**, 3601-3612.

3. P.C. Hansen (1987) Rank-deficient and discrete ill-posed problems: 
Numerical aspects of linear inversion (monographs on mathematical modeling and 
computation). *Society for Industrial and Applied Mathematics*.

Module Reference
----------------
.. automodule:: rampedpyrox.core.laplacetransform
	:members:








