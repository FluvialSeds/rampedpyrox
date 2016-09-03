'''
This module contains the Model superclass and all corresponding subclasses.

* TODO: Add ``from_ratedata`` classmethod once rate data classes have been 
defined!
'''

import matplotlib.pyplot as plt
import numpy as np

#import container classes
from rampedpyrox.core.array_classes import(
	rparray
	)

#import other rampedpyrox classes
from rampedpyrox.timedata.timedata import(
	RpoThermogram,
	)

from rampedpyrox.ratedata.ratedata import(
	EnergyComplex,
	)

#import helper functions
from rampedpyrox.model.model_helper import(
	_calc_phi,
	_rpo_calc_A,
	)

class Model(object):
	'''
	Class to store model setup. Intended for subclassing, do not call
	directly.
	'''

	def __init__(self, A, model_type, t, T):
		'''
		Initialize the superclass.

		Parameters
		----------
		A : 2d array-like
			Array of the transform matrix to convert from time to rate space.
			Rows are timepoints and columns are rates. Shape [nt x nk].

		model_type : str
			A string of the model type

		t : array-like
			Array of time, in seconds. Length nt.

		T : scalar or array-like
			Scalar or array of temperature, in Kelvin.
		'''

		#ensure data is in the right form
		nt = len(t)
		t = rparray(t, nt)
		T = rparray(T, nt)
		A = rparray(A, nt)

		#store attributes
		self.A = A
		self.model_type = model_type
		self.nt = nt
		self.t = t
		self.T = T

	#define a class method for creating instance directly from timedata
	@classmethod
	def from_timedata(self):
		raise NotImplementedError

	#define a class method for creating instance directly from ratedata
	@classmethod
	def from_ratedata(self):
		raise NotImplementedError

	#define a method for calculating the L curve
	def calc_L_curve(self, timedata, ax=None, plot=False, **kwargs):
		'''
		Function to calculate the L-curve for a given model and timedata
		instance in order to choose the best-fit smoothing parameter, omega.

		Parameters
		----------
		timedata : rp.TimeData
			Instance of ``TimeData`` subclass containing the time and fraction
			remaining arrays to use in L curve calculation.

		Keyword Arguments
		-----------------
		ax : None or matplotlib.axis
			Axis to plot on. If `None` and ``plot=True``, automatically 
			creates a ``matplotlip.axis`` instance to return. Defaults to 
			`None`.

		plot : Boolean
			Tells the method to plot the resulting L curve or not.

		om_min : int
			Minimum omega value to search. Defaults to 1e-3.

		om_max : int
			Maximum omega value to search. Defaults to 1e2.

		nOm : int
			Number of omega values to consider. Defaults to 150.

		Returns
		-------
		om_best : float
			The calculated best-fit omega value.

		axis : None or matplotlib.axis
			If ``plot=True``, returns an updated axis handle with plot.
		
		Notes
		-----

		See Also
		--------
		plot_L_curve
			Package method for ``plot_L_curve``.

		References
		----------
		D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
		respiration rates from decay time series. *Biogeosciences*, **9**,
		3601-3612.

		P.C. Hansen (1987) Rank-deficient and discrete ill-posed problems:
		Numerical aspects of linear inversion (monographs on mathematical
		modeling and computation). *Society for Industrial and Applied
		Mathematics*.

		P.C. Hansen (1994) Regularization tools: A Matlab package for analysis and
		solution of discrete ill-posed problems. *Numerical Algorithms*, **6**,
		1-35.
		'''

		#pop acceptable kwargs:
		#	om_min
		#	om_max
		#	nOm

		om_min = kwargs.pop('om_min', 1e-3)
		om_max = kwargs.pop('om_max', 1e2)
		nOm = kwargs.pop('nOm', 150)

		if kwargs:
			raise TypeError(
				'Unexpected **kwargs: %r' % kwargs)

		#define arrays
		log_om_vec = np.linspace(np.log10(om_min),np.log10(om_max),nOm)
		om_vec = 10**rparray(log_om_vec, nOm)

		res_vec = []; rgh_vec = []

		#for each omega value in the vector, calculate the errors
		for i, w in enumerate(om_vec):
			_, res, rgh = _calc_phi(self, timedata, w)
			res_vec.append(res)
			rgh_vec.append(rgh)

		#store logs as rparrays, removing noise after 6 sig figs
		res_vec = rparray(np.log10(res_vec), nOm, sig_figs=6)
		rgh_vec = rparray(np.log10(rgh_vec), nOm, sig_figs=6)

		#calculate derivatives and curvature
		dydx = rgh_vec.derivatize(res_vec)
		dy2d2x = dydx.derivatize(res_vec)

		k = np.abs(dy2d2x)/(1+dydx**2)**1.5

		#find first occurrance of argmax k
		i = np.argmax(k)
		om_best = om_vec[i]

		#plot if necessary
		if plot:

			#create axis if necessary
			if ax is None:
				_,ax = plt.subplots(1,1)

			#plot results
			ax.plot(res_vec,rgh_vec,
				linewidth=2,
				color='k',
				label='L-curve')

			ax.scatter(res_vec[i],rgh_vec[i],
				s=50,
				facecolor='w',
				edgecolor='k',
				linewidth=1.5,
				label=r'best-fit $\omega$')

			#set axis labels and text
			ax.set_xlabel(r'$log_{10} \parallel A\phi - g \parallel$')
			ax.set_ylabel(r'$log_{10} \parallel R\phi \parallel$')

			label1 = r'best-fit $\omega$ = %.3f' %(om_best)
			label2 = r'$log_{10} \parallel A\phi - g \parallel$ = %.3f' \
				%(res_vec[i])
			label3 = r'$log_{10} \parallel R\phi \parallel$  = %0.3f' \
				%(rgh_vec[i])

			ax.text(0.5,0.95,label1+'\n'+label2+'\n'+label3,
				verticalalignment='top',
				horizontalalignment='left',
				transform=ax.transAxes)

			return om_best, ax

		else:
			return om_best


class LaplaceTransform(Model):
	'''
	Class to store Laplace transform model setup. Intended for subclassing,
	do not call directly.
	'''

	def __init__(self, A, model_type, t, T):

		super(LaplaceTransform, self).__init__(A, model_type, t, T)

	@classmethod
	def from_timedata(self):
		raise NotImplementedError

	@classmethod
	def from_ratedata(self):
		raise NotImplementedError


class Daem(LaplaceTransform):
	__doc__='''
	Class to calculate the `DAEM` model Laplace Transform. Used for ramped-
	temperature kinetic problems such as Ramped PyrOx, pyGC, TGA, etc.
	
	Parameters
	----------
	Ea : array-like
		Array of Ea values, in kJ/mol. Length nE.

	log10k0 : scalar, array-like, or lambda function
		Arrhenius pre-exponential factor, either a constant value, array-like
		with length nEa, or a lambda function of Ea. 

	t : array-like
		Array of time, in seconds. Length nt.

	T : array-like
		Array of temperature, in Kelvin. Length nt.

	Raises
	------
	TypeError
		If `t` is not array-like.

	TypeError
		If `Ea` is not scalar or array-like.

	ValueError
		If `T` is not scalar or array-like with length nt.

	ValueError
		If log10k0 is not scalar, lambda, or array-like with length nEa.

	Warnings
	--------
	If attempting to use isothermal data to create a ``Daem`` model
	instance.

	Notes
	-----
	Best-fit omega values using the L-curve approach typically under-
	regularize for Ramped PyrOx data. That is, `om_best` calculated here
	results in a "peakier" f(Ea) and a higher number of Ea Gaussian peaks
	than can be resolved given a typical run with ~5-7 CO2 fractions. Omega
	values between 1 and 5 typically result in ~5 Ea Gaussian peaks for most
	Ramped PyrOx samples.

	See Also
	--------
	RpoThermogram
		``TimeData`` subclass for storing and analyzing RPO time/temp. data.

	EnergyComplex
		``RateData`` subclass for storing, deconvolving, and analyzing RPO
		rate data.

	Examples
	--------
	Creating a DAEM using manually-inputted Ea, k0, t, and T::

		#import modules
		import numpy as np
		import rampedpyrox as rp

		#generate arbitrary data
		t = np.arange(1,100) #100 second experiment
		beta = 0.5 #K/second
		T = beta*t + 273.15 #K
		Ea = np.arange(50, 350) #kJ/mol
		log10k0 = 10 #seconds-1

		#create instance
		daem = rp.Daem(Ea, log10k0, t, T)

	Creating a DAEM from real thermogram data using the ``Daem.from_timedata``
	class method::

		#import modules
		import rampedpyrox as rp

		#create thermogram instance
		tg = rp.RpoThermogram.from_csv('some_data_file.csv')

		#create Daem instance
		daem = rp.Daem.from_timedata(tg, 
			Ea_max=350, 
			Ea_min=50, 
			nEa=250, 
			log10k0=10)

	Creating a DAEM from an energy complex using the
	``Daem.from_ratedata`` class method::

		#import modules
		import rampedpyrox as rp

		#create energycomplex instance
		ec = rp.EnergyComplex(Ea, fEa)

		#create Daem instance
		daem = rp.Daem.from_ratedata(ec, 
			beta=0.08, 
			log10k0=10, 
			nt=250, 
			t0=0, 
			T0=373, 
			tf=1e4)

	Plotting the L-curve of a Daem to find the best-fit omega value::

		#import modules
		import matplotlib.pyplot as plt

		#create figure
		fig, ax = plt.subplots(1,1)

		#plot L curve
		om_best, ax = daem.calc_L_curve(tg,
			ax=None, 
			plot=False,
			om_min = 1e-3,
			om_max = 1e2,
			nOm = 150)

	Attributes
	----------
	A : rp.rparray

	Ea : rp.rparray
		Array of Ea values, in kJ/mol. Length nE.

	nEa : int
		Number of activation energy points.

	nt : int
		Number of timepoints.

	t : rp.rparray
		Array of timep, in seconds. Length nt.

	T : rp.rparray
		Array of temperature, in Kelvin. Length nt.

	References
	----------
	R.L Braun and A.K. Burnham (1987) Analysis of chemical reaction kinetics
	using a distribution of activation energies and simpler models.
	*Energy & Fuels*, **1**, 153-161.

	\B. Cramer et al. (1998) Modeling isotope fractionation during primary
	cracking of natural gas: A reaction kinetic approach. *Chemical
	Geology*, **149**, 235-250.

	\V. Dieckmann (2005) Modeling petroleum formation from heterogeneous
	source rocks: The influence of frequency factors on activation energy
	distribution and geological prediction. *Marine and Petroleum Geology*,
	**22**, 375-390.

	D.C. Forney and D.H. Rothman (2012) Common structure in the
	heterogeneity of plant-matter decay. *Journal of the Royal Society
	Interface*, rsif.2012.0122.

	D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
	respiration rates from decay time series. *Biogeosciences*, **9**,
	3601-3612.

	P.C. Hansen (1987) Rank-deficient and discrete ill-posed problems:
	Numerical aspects of linear inversion (monographs on mathematical
	modeling and computation). *Society for Industrial and Applied
	Mathematics*.

	P.C. Hansen (1994) Regularization tools: A Matlab package for analysis and
	solution of discrete ill-posed problems. *Numerical Algorithms*, **6**,
	1-35.

	J.E. White et al. (2011) Biomass pyrolysis kinetics: A comparative
	critical review with relevant agricultural residue case studies.
	*Journal of Analytical and Applied Pyrolysis*, **91**, 1-33.
	'''

	def __init__(self, Ea, log10k0, t, T):

		#warn if T is scalar
		if isinstance(T, (int, float)):
			warnings.warn((
				"Attempting to use isothermal data to create DAEM model! T is"
				"a scalar value of: %.1f. Consider using an isothermal model" 
				"type instead." % T))


		#calculate A matrix
		A = _rpo_calc_A(Ea, log10k0, t, T)
		model_type = 'Daem'

		super(Daem, self).__init__(A, model_type, t, T)

		#store Daem-specific attributes
		nEa = len(Ea)
		self.log10k0 = log10k0
		self.Ea = rparray(Ea, nEa)
		self.nEa = nEa

	@classmethod
	def from_timedata(cls, timedata, Ea_max=350, Ea_min=50, log10k0=10, 
		nEa=250):
		'''
		Class method to directly generate a ``Daem`` instance using data
		stored in a ``TimeData`` instance.

		Parameters
		----------
		timedata : rp.TimeData
			Instance of ``TimeData`` subclass containing the time and fraction
			remaining arrays to use in L curve calculation.

		Keyword Arguments
		-----------------
		Ea_max : int
			The maximum activation energy value to consider, in kJ/mol.

		Ea_min : int
			The minimum activation energy value to consider, in kJ/mol.

		log10k0 : scalar, array-like, or lambda function
			Arrhenius pre-exponential factor, either a constant value, array-
			likewith length nEa, or a lambda function of Ea. 
		
		nEa : int
			The number of activation energy points.

		Raises
		------
		ValueError
			If log10k0 is not scalar, lambda, or array-like with length nEa.

		Warnings
		--------
		If attempting to create a DAEM with an isothermal timedata instance.

		Notes
		-----

		See Also
		--------

		'''

		#check that timedata is the right type
		if not isinstance(timedata, (RpoThermogram)):
			warnings.warn((
			"Attempting to generate Daem model using a timedata instance of"
			"class: %s. Consider using RpoThermogram timedata instance"
			"instead." % repr(timedata)))

		#generate Ea, t, and T array
		Ea = np.linspace(Ea_min, Ea_max, nEa)
		t = timedata.t
		T = timedata.T

		return cls(Ea, log10k0, t, T)

	# @classmethod
	# def from_ratedata(cls, ratedata, beta=0.08, log10k0=10, nt=250, t0=0, 
	# 	T0=373, tf=1e4):
	# 	'''
	# 	Class method to directly generate a ``Daem`` instance using data
	# 	stored in a ``RateData`` instance.
	# 	'''

	# 	#check that ratedata is the right type
	# 	if not isinstance(ratedata, (EnergyComplex)):
	# 		raise TypeError((
	# 			"Attempting to generate Daem model using a ratedata instance"
	# 			"of class: %s. Only EnergyComplex instances can be used for"
	# 			"Daem models." % repr(ratedata)))

	# 	#generate Ea, t, and T array
	# 	Ea = ratedata.Ea
	# 	t = np.linspace(t0, tf, nt)
	# 	T = T0 + beta*t

	# 	return cls(Ea, log10k0, t, T)







