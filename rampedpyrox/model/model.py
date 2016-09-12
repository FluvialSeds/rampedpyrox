'''
This module contains the Model superclass and all corresponding subclasses.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['Daem']

import matplotlib.pyplot as plt
import numpy as np
import warnings

#import exceptions
from ..core.exceptions import(
	ScalarError,
	)

#import helper functions
from ..core.core_functions import(
	assert_len,
	derivatize,
	)

from .model_helper import(
	_calc_f,
	_rpo_calc_A,
	)

class Model(object):
	'''
	Class to store model setup. Intended for subclassing, do not call
	directly.
	'''

	def __init__(self, A, t, T):
		'''
		Initialize the superclass.

		Parameters
		----------
		A : 2d array-like
			Array of the transform matrix to convert from time to rate space.
			Rows are timepoints and columns are k/Ea values. Shape 
			[`nt` x `nk`].

		t : array-like
			Array of time, in seconds. Length `nt`.

		T : scalar or array-like
			Scalar or array of temperature, in Kelvin. If array, length `nt`.
		'''

		#ensure data is in the right form
		nt = len(t)
		t = assert_len(t, nt)
		T = assert_len(T, nt)
		A = assert_len(A, nt)

		#store attributes
		self.A = A
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
	def calc_L_curve(
			self, 
			timedata, 
			ax = None, 
			nOm = 150, 
			om_max = 1e2, 
			om_min = 1e-3, 
			plot = False):
		'''
		Function to calculate the L-curve for a given model and timedata
		instance in order to choose the best-fit smoothing parameter, omega.

		Parameters
		----------
		timedata : rp.TimeData
			``rp.TimeData`` instance containing the time and fraction
			remaining arrays to use in L curve calculation.

		ax : None or matplotlib.axis
			Axis to plot on. If `None` and ``plot = True``, automatically 
			creates a ``matplotlip.axis`` instance to return. Defaults to 
			`None`.

		nOm : int
			Number of omega values to consider. Defaults to 150.

		om_max : int
			Maximum omega value to search. Defaults to 1e2.

		om_min : int
			Minimum omega value to search. Defaults to 1e-3.

		plot : Boolean
			Tells the method to plot the resulting L curve or not. Defaults to
			`False`.

		Returns
		-------
		om_best : float
			The calculated best-fit omega value.

		axis : None or matplotlib.axis
			If ``plot = True``, returns an updated axis handle with plot.

		Raises
		------
		ScalarError
			If `om_max` or `om_min` are not scalar.

		ScalarError
			If `nOm` is not int.

		See Also
		--------
		plot_L_curve
			Package-level method for ``plot_L_curve``.

		References
		----------
		[1] D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
			respiration rates from decay time series. *Biogeosciences*, **9**,
			3601-3612.

		[2] P.C. Hansen (1987) Rank-deficient and discrete ill-posed problems:
			Numerical aspects of linear inversion (monographs on mathematical
			modeling and computation). *Society for Industrial and Applied*
			*Mathematics*.

		[3] P.C. Hansen (1994) Regularization tools: A Matlab package for analysis and
			solution of discrete ill-posed problems. *Numerical Algorithms*, **6**,
			1-35.
		'''

		#check that nOm, om_max, and om_min are in the right form
		if not isinstance(om_max, (int, float)):
			raise ScalarError(
				'om_max must be float or int')

		elif not isinstance(om_min, (int, float)):
			raise ScalarError(
				'om_min must be float or int')

		elif not isinstance(nOm, int):
			raise ScalarError(
				'nOm must be int')

		#define arrays
		log_om_vec = np.linspace(np.log10(om_min), np.log10(om_max), nOm)
		om_vec = 10**log_om_vec

		res_vec = np.zeros(nOm)
		rgh_vec = np.zeros(nOm)

		#for each omega value in the vector, calculate the errors
		for i, w in enumerate(om_vec):
			_, res, rgh = _calc_f(self, timedata, w)
			res_vec[i] = res
			rgh_vec[i] = rgh

		#store logs as arrays, and remove noise after 6 sig figs
		res_vec = np.log10(res_vec)
		rgh_vec = np.log10(rgh_vec)

		res_vec = np.around(res_vec, decimals = 6)
		rgh_vec = np.around(rgh_vec, decimals = 6)

		#calculate derivatives and curvature
		dydx = derivatize(rgh_vec, res_vec)
		dy2d2x = derivatize(dydx, res_vec)

		#function for curvature
		k = np.abs(dy2d2x)/(1+dydx**2)**1.5

		#find first occurrance of argmax k, ignoring first and last points
		i = np.argmax(k[1:-1])
		om_best = om_vec[i + 1]

		#plot if necessary
		if plot:

			#create axis if necessary
			if ax is None:
				_,ax = plt.subplots(1, 1)

			#plot results
			ax.plot(
				res_vec,
				rgh_vec,
				linewidth=2,
				color='k',
				label='L-curve')

			ax.scatter(
				res_vec[i],
				rgh_vec[i],
				s=50,
				facecolor='w',
				edgecolor='k',
				linewidth=1.5,
				label=r'best-fit $\omega$')

			#set axis labels and text
			ax.set_xlabel(
				r'residual rmse, $log_{10} \parallel Af - g \parallel$')
			
			ax.set_ylabel(
				r'roughness rmse, $log_{10} \parallel Rf \parallel$')

			label1 = r'best-fit $\omega$ = %.3f' %(om_best)
			
			label2 = (
				r'$log_{10} \parallel Af - g \parallel$ = %.3f' %(res_vec[i]))
			
			label3 = (
				r'$log_{10} \parallel Rf \parallel$  = %0.3f' %(rgh_vec[i]))

			ax.text(
				0.5,
				0.95,
				label1 + '\n' + label2 + '\n' + label3,
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

	def __init__(self, A, t, T):

		super(LaplaceTransform, self).__init__(A, t, T)

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
		Array of Ea values, in kJ/mol. Length `nEa`.

	log10k0 : scalar, array-like, or lambda function
		Arrhenius pre-exponential factor, either a constant value, array-like
		with length `nEa`, or a lambda function of Ea. 

	t : array-like
		Array of time, in seconds. Length `nt`.

	T : array-like
		Array of temperature, in Kelvin. Length `nt`.

	Warnings
	--------
	UserWarning
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
		``rp.TimeData`` subclass for storing and analyzing RPO 
		time/temperature data.

	EnergyComplex
		``rp.RateData`` subclass for storing, deconvolving, and analyzing RPO
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

	Creating a DAEM from real thermogram data using the ``rp.Daem.from_timedata``
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
	``rp.Daem.from_ratedata`` class method::

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

	**Attributes**

	A : np.ndarray

	Ea : np.ndarray
		Array of Ea values, in kJ/mol. Length `nEa`.

	nEa : int
		Number of activation energy points.

	nt : int
		Number of timepoints.

	t : np.ndarray
		Array of timep, in seconds. Length `nt`.

	T : np.ndarray
		Array of temperature, in Kelvin. Length `nt`.

	References
	----------
	[1] R.L Braun and A.K. Burnham (1987) Analysis of chemical reaction 
		kinetics using a distribution of activation energies and simpler 
		models. *Energy & Fuels*, **1**, 153-161.

	[2] B. Cramer et al. (1998) Modeling isotope fractionation during primary
		cracking of natural gas: A reaction kinetic approach. *Chemical*
		*Geology*, **149**, 235-250.

	[3] V. Dieckmann (2005) Modeling petroleum formation from heterogeneous
		source rocks: The influence of frequency factors on activation energy
		distribution and geological prediction. *Marine and Petroleum*
		*Geology*, **22**, 375-390.

	[4] D.C. Forney and D.H. Rothman (2012) Common structure in the
		heterogeneity of plant-matter decay. *Journal of the Royal Society*
		*Interface*, rsif.2012.0122.

	[5] D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
		respiration rates from decay time series. *Biogeosciences*, **9**,
		3601-3612.

	[6] P.C. Hansen (1987) Rank-deficient and discrete ill-posed problems:
		Numerical aspects of linear inversion (monographs on mathematical
		modeling and computation). *Society for Industrial and Applied*
		*Mathematics*.

	[7] P.C. Hansen (1994) Regularization tools: A Matlab package for analysis
		and solution of discrete ill-posed problems. *Numerical Algorithms*, 
		**6**, 1-35.

	[8] C.C. Lakshmananan et al. (1991) Implications of multiplicity in
		kinetic parameters to petroleum exploration: Distributed activation
		energy models. *Energy & Fuels*, **5**, 110-117.

	[9] J.E. White et al. (2011) Biomass pyrolysis kinetics: A comparative
		critical review with relevant agricultural residue case studies.
		*Journal of Analytical and Applied Pyrolysis*, **91**, 1-33.
	'''

	def __init__(self, Ea, log10k0, t, T):

		#warn if T is scalar
		if isinstance(T, (int, float)):
			warnings.warn(
				'Attempting to use isothermal data for RPO run! T is a scalar'
				'value of: %r. Consider using an isothermal model type'
				'instead.' % T, UserWarning)

		elif len(set(T)) == 1:
			warnings.warn(
				'Attempting to use isothermal data for RPO run! T is a scalar'
				'value of: %r. Consider using an isothermal model type'
				'instead.' % T[0], UserWarning)

		#get log10k0 into the right format
		if hasattr(log10k0,'__call__'):
			log10k0 = log10k0(Ea)

		#calculate A matrix
		A = _rpo_calc_A(Ea, log10k0, t, T)

		super(Daem, self).__init__(A, t, T)

		#store Daem-specific attributes
		nEa = len(Ea)
		self.log10k0 = assert_len(log10k0, nEa)
		self.Ea = assert_len(Ea, nEa)
		self.nEa = nEa

	@classmethod
	def from_timedata(
			cls, 
			timedata, 
			Ea_max = 350, 
			Ea_min = 50, 
			log10k0 = 10, 
			nEa = 250):
		'''
		Class method to directly generate an ``rp.Daem`` instance using data
		stored in an ``rp.TimeData`` instance.

		Parameters
		----------
		timedata : rp.TimeData
			``rp.TimeData`` instance containing the time array to use
			for creating the DAEM.

		Ea_max : int
			The maximum activation energy value to consider, in kJ/mol.
			Defaults to 350.

		Ea_min : int
			The minimum activation energy value to consider, in kJ/mol.
			Defaults to 50.

		log10k0 : scalar, array-like, or lambda function
			Arrhenius pre-exponential factor, either a constant value, array-
			likewith length `nEa`, or a lambda function of Ea. Defaults to 10.
		
		nEa : int
			The number of activation energy points. Defaults to 250.

		Warnings
		--------
		UserWarning
			If attempting to create a DAEM with an isothermal timedata 
			instance.

		See Also
		--------
		from_ratedata
			Class method to directly generate an ``rp.Daem`` instance using 
			data stored in an ``rp.RateData`` instance.
		'''

		#warn if timedata is not RpoThermogram
		td_type = type(timedata).__name__

		if td_type not in ['RpoThermogram']:
			warnings.warn(
				'Attempting to calculate isotopes using an isothermal timedata'
				' instance of type %r. Consider using rp.RpoThermogram' 
				' instance instead' % td_type, UserWarning)

		#generate Ea, t, and T array
		Ea = np.linspace(Ea_min, Ea_max, nEa)
		t = timedata.t
		T = timedata.T

		return cls(Ea, log10k0, t, T)

	@classmethod
	def from_ratedata(
			cls, 
			ratedata, 
			beta = 0.08, 
			log10k0 = 10, 
			nt = 250,
			t0 = 0, 
			T0 = 373, 
			tf = 1e4):
		'''
		Class method to directly generate an ``rp.Daem`` instance using data
		stored in an ``rp.RateData`` instance.

		Paramters
		---------
		ratedata : rp.RateData
			``rp.RateData`` instance containing the Ea array to use for
			creating the DAEM. 

		beta : int or float
			Temperature ramp rate to use in model, in Kelvin/second. Defaults
			to 0.08 (*i.e.* 5K/min)

		log10k0 : scalar, array-like, or lambda function
			Arrhenius pre-exponential factor, either a constant value, array-
			likewith length `nEa`, or a lambda function of Ea. Defaults to 10.

		nt : int
			The number of time points to use. Defaults to 250.

		t0 : int or float
			The initial time to be used in the model, in seconds. Defaults 
			to 0.

		T0 : int or float
			The initial temperature to be used in the model, in Kelvin.
			Defaults to 373.

		tf : int or float
			The final time to be used in the model, in seconds. Defaults to
			10,000.

		See Also
		--------
		from_timedata
			Class method to directly generate an ``rp.Daem`` instance using data
			stored in an ``rp.TimeData`` instance.

		'''

		#warn if ratedata is not EnergyComplex
		rd_type = type(ratedata).__name__

		if rd_type not in ['EnergyComplex']:
			warnings.warn(
				'Attempting to calculate isotopes using a ratedata instance of'
				' type %r. Consider using rp.EnergyComplex instance instead'
				% rd_type, UserWarning)

		#generate Ea, t, and T array
		Ea = ratedata.Ea
		t = np.linspace(t0, tf, nt)
		T = T0 + beta*t

		return cls(Ea, log10k0, t, T)

if __name__ == '__main__':

	import rampedpyrox as rp