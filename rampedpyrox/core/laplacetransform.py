'''
``laplacetransform`` module for calculating the Laplace Transform for a given 
model and performing the inverse/forward transformation. Stores information 
as a ``LaplaceTransform`` intstance.

* TODO: update calc_L_curve to be more pythonic.
* TODO: Keep testing "calc_Wabha" function and developing best-fit omega
'''

from __future__ import print_function

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import nnls
from numpy.linalg import norm
from numpy.linalg import inv

from rampedpyrox.core.thermogram import ModeledData

__docformat__ = 'restructuredtext en'

## PRIVATE FUNCTIONS ##

#define function to calculate phi
def _calc_phi(A, g, omega):
	'''
	Calculates `phi`, the discretized Lapalce transform f(Ea) result.
	Called by ``LaplaceTransform.calc_EC_inv()``.
	Called by ``calc_L()``.

	Parameters
	----------
	A : np.ndarray
		Laplace Transform matrix of shape [nT x nE].

	g : np.ndarray
		Array of fraction of carbon remaining.

	omega : int
		Omega value for Tikhonov Regularization.

	Returns
	-------
	phi : np.ndarray
		Array of the pdf of the distribution of Ea, f(Ea).

	resid_err : float
		Residual RMSE between true and modeled thermogram.

	rgh_err : float
		Roughness RMSE from Tikhonov Regularization.
	'''

	#extract scalars, calculate R
	nT,nE = np.shape(A)
	R = _calc_R(nE)

	#concatenate A,R and g,zeros
	A_reg = np.concatenate((A,R*omega))
	g_reg = np.concatenate((g,np.zeros(nE+1)))

	#calculate inverse results and errors
	phi,_ = nnls(A_reg,g_reg)
	g_hat = np.inner(A,phi)

	resid_err = norm(g-g_hat)/nT
	rgh = np.inner(R,phi)
	rgh_err = norm(rgh)/nE

	return phi, resid_err, rgh_err

#define function to calculate R matrix
def _calc_R(n):
	'''
	Calculates regularization matrix (`R`) for a given size, n.
	Called by ``_calc_phi()``.

	Parameters
	----------
	n : int
		Size of the regularization matrix will be [n+1 x n].

	Returns
	-------
	R : np.ndarray
		Regularization matrix of size [n+1 x n].

	References
	----------
	D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
	respiration rates from decay time series. *Biogeosciences*, **9**,
	3601-3612.
	'''

	R = np.zeros([n+1,n])
	R[0,0] = 1.0 #ensure pdf = 0 outside of Ea range specified
	R[-1,-1] = -1.0

	c = [-1,1] #1st derivative operator

	for i,row in enumerate(R):
		if i != 0 and i != n:
			row[i-1:i+1] = c

	return R

#function to round to sig fig for removing noise in L curve calculation
def _round_to_sigfig(vec, sig_figs=6):
	'''
	Rounds inputted vector to specified sig fig.
	Called by ``calc_L_curve``.

	Parameters
	----------
	vec : np.ndarray
		Array of data to round.

	sig_figs : int
		Number of sig figs to round to. Defaults to 6.

	Returns
	-------
	vec_round : np.ndarray
		`vec` array rounded to `sig_figs`.

	References
	----------
	D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
	respiration rates from decay time series. *Biogeosciences*, **9**,
	3601-3612.
	'''

	p = sig_figs
	order = np.floor(np.log10(vec))
	vecH = 10**(p-order-1)*vec
	vec_rnd_log = np.round(vecH)
	vec_round = vec_rnd_log/10**(p-order-1)

	return vec_round

## PUBLIC FUNCTIONS ##

#define function to calculate A matrix for ``LaplaceTransform``
def calc_A(t, Tau, eps, logk0):
	'''
	Calculates the Laplace transform matrix A assuming a first-order DAEM.
	Called by ``rp.LaplaceTransform.__init__()``.

	Parameters
	----------
	t : np.ndarray
		Array of timepoints (in seconds) from ``rp.RealData`` instance,
		length nT.

	Tau : np.ndarray
		Array of temperature points (in Kelvin) from ``rp.RealData`` instance,
		length nT.

	eps : np.ndarray
		Array of Ea values (in kJ/mol), length nE.

	logk0 : np.ndarray
		Array of Arrhenius pre-exponential factor values of length nE. See 
		White et al. (2011) for a review on logk0 values in biomass and 
		Dieckmann (2005) for a discussion on logk0 values in petroleum 
		formations.

	Returns
	-------
	A : np.ndarray
		Laplace Transform matrix of shape [nT x nE].

	Raises
	------
	ValueError
		If `t` and `Tau` arrays are not the same length.

	ValueError
		If `eps` and `logk0` arrays are not the same length.

	Notes
	-----
	The `A` matrix does not depend on the actual thermogram, only on the time-
	temperature history and the pre-exponential factor `logk0`. That is, two
	thermograms with the same time-temperature history will generate the same
	`A` matrix.

	Time-temperature history need not be constant. ``calc_A`` allows for non-
	constant ramp rates, including isothermal conditions.

	See Also
	--------
	calc_L_curve
	LaplaceTransform

	Examples
	--------
	Calculating an A matrix for given input data::

		#t and Tau from rp.RealData instance rd
		eps = np.arange(50,350) #Ea range to calculate over
		logk0 = 10 #constant pre-exponential (Arrhenius) factor

		A = rp.calc_A(rd.t, rd.Tau, eps, logk0)

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

	J.E. White et al. (2011) Biomass pyrolysis kinetics: A comparative
	critical review with relevant agricultural residue case studies.
	*Journal of Analytical and Applied Pyrolysis*, **91**, 1-33.
	'''

	#set constants
	nT = len(t)
	nE = len(eps)
	k0 = 10**logk0 #s-1
	R = 8.314/1000 #kJ/mol/K
	del_t = t[1]-t[0] #s
	del_eps = eps[1]-eps[0] #kJ

	#check array lengths
	if len(Tau) != nT:
		raise ValueError('t and Tau must be same length.')
	elif len(k0) != nE:
		raise ValueError('eps and logk0 must be same length.')

	#pre-allocate A
	A = np.zeros([nT,nE])

	#loop through each timestep
	for i,_ in enumerate(Tau):

		#generate matrices
		U = Tau[:i] #Kelvin, [1,i]
		eps_mat = np.outer(eps,np.ones(i)) #kJ, [nE,i]
		k0_mat = np.outer(k0,np.ones(i)) #s-1, [nE,i]
		u_mat = np.outer(np.ones(nE),U) #Kelvin, [nE,i]

		#generate A for row i (i.e. for each timepoint) and store in A
		hE_mat = -k0_mat*del_t*np.exp(-eps_mat/(R*u_mat)) #unitless, [nE,i]
		A[i] = np.exp(np.sum(hE_mat,axis=1))*del_eps #kJ

	return A

#define function to calculate L curve
def calc_L_curve(A, g, log_om_min=-3, log_om_max=2, nOm=100):
	'''
	Calculates the L curve for a given matrix `A` and fraction of carbon
	remaining `g`.
	Called by ``rp.LaplaceTransform.calc_EC_inv()`` if omega is 'auto'.
	Called by ``rp.LaplaceTransform.plot_L_curve()``.

	Parameters
	----------
	A : np.ndarray
		Laplace Transform matrix of shape [nT x nE].

	g : np.ndarray
		Array of fraction of carbon remaining.

	log_om_min : int
		Log10 of minimum omega value to search. Defaults to -3.

	log_om_max : int
		Log10 of maximum omega value to search. Defaults to 2.

	nOm : int
		Number of omega values to consider. Defaults to 100.

	Returns
	-------
	om_best : float
		Omega value at the point of maximum curvature in the L-curve plot of
		resididual RMSE vs. roughness RMSE. See Hansen (1987; 1994) for 
		discussion on `om_best` calculation.

	resid_vec : np.ndarray
		Array of log10 of the residual RMSE for each omega value considered,
		length `nOm`.
	
	rgh_vec : np.ndarray
		Array of log10 of the roughness RMSE for each omega value considered,
		length `nOm`.
	
	omega_vec : np.ndarray
		Array of the omega values considered, length `nOm`.

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
	calc_A
	LaplaceTransform

	Examples
	--------
	Basic implementation::

		#assuming A matrix calculated by rp.calc_A and g from rp.RealData 
		# instance rd.

		om_best,resid_vec,rgh_vec,omega_vec = rp.calc_L_curve(A,g)

	References
	----------
	D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
	respiration rates from decay time series. *Biogeosciences*, **9**,
	3601-3612.

	P.C. Hansen (1987) Rank-deficient and discrete ill-posed problems:
	Numerical aspects of linear inversion (monographs on mathematical modeling
	and computation). *Society for Industrial and Applied Mathematics*.

	P.C. Hansen (1994) Regularization tools: A Matlab package for analysis and
	solution of discrete ill-posed problems. *Numerical Algorithms*, **6**,
	1-35.
	'''

	#shape of A is nT x nE
	nT,nE = np.shape(A)

	#make omega_vec and pre-allocate error vectors
	log_om_vec = np.linspace(log_om_min,log_om_max,nOm)
	omega_vec = 10**(log_om_vec)
	resid_vec = np.zeros(nOm)
	rgh_vec = np.zeros(nOm)

	#for each omega value in the vector, calculate the errors
	for i,w in enumerate(omega_vec):
		_,res,rgh = _calc_phi(A,g,w)
		resid_vec[i] = res
		rgh_vec[i] = rgh

	#remove noise in the L-curve beyond 6 sig figs
	resid_vec = _round_to_sigfig(resid_vec,sig_figs=6)
	rgh_vec = _round_to_sigfig(rgh_vec,sig_figs=6)

	#transform to log space
	resid_vec = np.log10(resid_vec)
	rgh_vec = np.log10(rgh_vec)

	#calculate derivatives
	dres = np.diff(resid_vec)
	dres1 = dres[:-1]
	dres2 = dres[1:]

	drgh = np.diff(rgh_vec)
	drgh1 = drgh[:-1]
	drgh2 = drgh[1:]

	dom = np.diff(omega_vec)
	dom1 = dom[:-1]
	dom2 = dom[1:]    

	#calculate derivative at om_n, the omega value at the center of the 1st
	#and 2nd derivatives
	om_n = omega_vec[1:-1]/2 + (omega_vec[:-2]+omega_vec[2:])/4

	#first derivative at om_n
	drgh_n=(drgh1/dom1+drgh2/dom2)/2
	dres_n=(dres1/dom1+dres2/dom2)/2

	#second derivative at om_n
	ddrgh_n=(drgh2/dom2-drgh1/dom1)/(dom1/2+dom2/2)
	ddres_n=(dres2/dom2-dres1/dom1)/(dom1/2+dom2/2)

	#calculate curvature and find maximum
	curv = (dres_n*ddrgh_n - ddres_n*drgh_n)/(dres_n**2 + drgh_n**2)**1.5

	max_curv = np.argmax(curv)
	om_best = omega_vec[max_curv]

	return om_best, resid_vec, rgh_vec, omega_vec 

def calc_Wabha(A, g, omega):
	__doc__='''
	Currently unused. Testing alternate methods of choosing best-fit omega.

	Parameters
	----------
	A : np.ndarray
		Laplace Transform matrix of shape [nT x nE].

	g : np.ndarray
		Array of fraction of carbon remaining.

	omega : int or float
		Omega value for regularization.

	References
	----------
	\G. Wabha (1980) Spline models for observational data (monographs on 
	mathematical modeling and computation). *Society for Industrial and*
	*Applied Mathematics*.
	'''


	#extract shape
	nT,nE = np.shape(A)

	#calculate phi and g_hat
	phi,_,_ = _calc_phi(A,g,omega)
	g_hat = np.inner(A,phi)

	#calculate RSS
	RSS = norm(g-g_hat)**2

	#calculate d.o.f., tau
	x = inv(np.dot(A.T,A) + (omega**2)*np.eye(nE))
	X = np.dot(np.dot(A,x),A.T)
	tau = np.trace(np.eye(nT) - X)**2

	return RSS/tau
	

class LaplaceTransform(object):
	__doc__='''
	Class for storing the A matrix and calculating forward/inverse results.

	Parameters
	----------
	t : np.ndarray
		Array of timepoints (in seconds) from ``rp.RealData`` instance,
		length nT.

	Tau : np.ndarray
		Array of temperature points (in Kelvin) from ``rp.RealData`` instance,
		length nT.

	eps : np.ndarray
		Array of Ea values (in kJ/mol), length nE.

	logk0 : int, float, or lambda
		Arrhenius pre-exponential factor, either a constant value or a lambda
		function of Ea. See White et al. (2011) for a review on logk0 values
		in biomass and Dieckmann (2005) for a discussion on logk0 values in
		petroleum formations.

	Raises
	------
	ValueError
		If `logk0` is not scalar, lambda function, or array of length nE.

	ValueError
		If `t` and `Tau` arrays do not have the same length.

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
	calc_A
	calc_L_curve

	Examples
	--------
	Calculating the Laplace Transform object::

		#load modules
		import numpy as np

		eps = np.arange(50,350) #Ea range to calculate over
		logk0 = 10. #pre-exponential (Arrhenius) factor
		
		lt = rp.LaplaceTransform(rd.t,rd.Tau,eps,logk0)
		
	Plotting the L-curve::

		#load modules
		import matplotlib.pyplot as plt
		fig,ax = plt.subplots(1,1)

		omega,ax = lt.plot_L_curve(rd, ax=ax)

	Calculating an f(Ea) distribution for an ``rp.RealData`` instance rd 
	using the inverse model::

		#calculate best-fit omega using the L-curve
		phi,resid_err,rgh_err,om_best = lt.calc_EC_inv(rd,omega='auto')

	Creating an ``rp.ModeledData`` instance of the forward-modeled thermogram
	using an ``rp.EnergyComplex`` instance ec::

		rd = lt.calc_TG_fwd(ec)

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

	def __init__(self, t, Tau, eps, logk0):

		#convert logk0 to array
		if hasattr(logk0,'__call__'):
			#checking if lambda function
			self._logk0 = logk0(eps) #kJ
		elif isinstance(logk0,(int,float)):
			#checking if scalar
			self._logk0 = np.ones(len(eps))*logk0 #kJ
		elif len(logk0) != len(eps):
			#raise error
			raise ValueError('logk0 must be lambda, scalar, or array of len nE')

		#calculate A matrix
		A = calc_A(t, Tau, eps, logk0)

		#define public parameters
		self.A = A #[nT,nE]
		self.t = t #seconds
		self.Tau = Tau #Kelvin
		self.eps = eps #kJ

	def calc_EC_inv(self, tg, omega='auto'):
		'''
		Calculates the Energy Complex for a given thermogram (inverse model).

		Parameters
		----------
		tg : rp.RealData
			``rp.RealData`` instance containing the thermogram to use for the
			inverse model.

		omega : int or str
			Omega value for Tikhonov regularization, either an integer or 'auto'.
			Defaults to 'auto'.

		Returns
		-------
		phi : np.ndarray
			Array of the pdf of the distribution of Ea, f(Ea).

		resid_err : float
			Residual RMSE between true and modeled thermogram.

		rgh_err : float
			Roughness RMSE from Tikhonov Regularization.

		om_best : float
			Omega value at the point of maximum curvature in the L-curve plot of
			resididual RMSE vs. roughness RMSE. See Hansen (1987; 1994) for 
			discussion on `om_best` calculation.

		Raises
		------
		ValueError
			If the size of the A matrix does not match that of the  data in tg.
			
		ValueError
			If inputted omega is not int, float, or 'auto'.
		'''

		#extract thermogram data
		g = tg.g
		nE = len(self.eps)
		nT = len(tg.Tau)

		#check that the shape of A matches the thermogram
		if np.shape(self.A) != (nT,nE):
			raise ValueError('Size of A does not match Thermogram nT')

		#check omega input and calculate best-fit value if 'auto'
		if omega is 'auto':
			#calculate omega if auto
			om_best,_,_,_ = calc_L_curve(self.A, g)
		elif isinstance(omega,(int,float)):
			#simply pass inputted omega value
			om_best = omega
		else:
			#raise error
			raise ValueError('omega must be int, float, or "auto"')

		phi, resid_err, rgh_err = _calc_phi(self.A, g, om_best)

		return phi, resid_err, rgh_err, om_best

	def calc_TG_fwd(self, ec):
		'''
		Calculates the Thermogram for a given EnergyComplex (forward model).

		Parameters
		----------
		ec : rp.EnergyComplex
			``rp.EnergyComplex`` instance containing the invsersion model Ea
			Gaussian peaks.

		Returns
		-------
		md : rp.ModeledData
			``rp.ModeledData`` instance containing the estimated thermogram 
			using the inversion model results.
		'''

		#calculate the estimated g
		g_hat = np.inner(self.A,ec.phi_hat) #fraction

		#calculate g for each peak
		gp = np.inner(self.A,ec.peaks.T) #fractions
		
		#pass into a ModeledData object and return
		return ModeledData(self.t, self.Tau, g_hat, gp)


	def plot_L_curve(self, tg, ax=None, log_om_min=-3, log_om_max=2, nOm=100):
		'''
		Calculates the L curve w.r.t. a given ``rp.RealData`` instance and 
		plots the result.

		Parameters
		----------
		tg : rp.RealData
			``rp.RealData`` instance containing the thermogram to use for the
			inverse model.

		ax : None or matplotlib.axis
			Axis to plot on. If `None`, automatically creates a
			``matplotlip.axis`` instance to return. Defaults to `None`.

		log_om_min : int
			Log10 of minimum omega value to search. Defaults to -3.

		log_om_max : int
			Log10 of maximum omega value to search. Defaults to 2.

		nOm : int
			Number of omega values to consider. Defaults to 100.

		Returns
		-------
		om_best : float
			Omega value at the point of maximum curvature in the L-curve plot
			of resididual RMSE vs. roughness RMSE. See Hansen (1987; 1994) for
			discussion on `om_best` calculation.

		ax : matplotlib.axis
			Updated axis instance with plotted data.
		'''

		#calculate L curve
		om_best,resid_vec,rgh_vec,omega_vec = calc_L_curve(self.A, tg.g, 
			log_om_min=log_om_min, 
			log_om_max=log_om_max, 
			nOm=nOm)

		if ax is None:
			_,ax = plt.subplots(1,1)

		#extract the resid_err and rgh_err at best-fit omega
		i = np.where(omega_vec == om_best)[0]
		resid = resid_vec[i]
		rgh = rgh_vec[i]

		ax.plot(resid_vec,rgh_vec)
		ax.scatter(resid,rgh)

		ax.set_xlabel(r'$log_{10} \parallel A\phi - g \parallel$')
		ax.set_ylabel(r'$log_{10} \parallel R\phi \parallel$')

		label1 = r'best-fit $\omega$ = %.3f' %(om_best)
		label2 = r'$log_{10} \parallel A\phi - g \parallel$ = %.3f' %(resid)
		label3 = r'$log_{10} \parallel R\phi \parallel$  = %0.3f' %(rgh)


		ax.text(0.5,0.95,label1+'\n'+label2+'\n'+label3,
			verticalalignment='top',
			horizontalalignment='left',
			transform=ax.transAxes)

		return om_best, ax
