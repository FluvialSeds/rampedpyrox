'''
Laplacetransform module for calculating the Laplace Transform for a given model
and performing the inverse/forward transformation. Stores information in a
LaplaceTransform object.

* TODO: update calc_L_curve to be more pythonic.
* TODO: Add summary method.
'''

import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import nnls
from numpy.linalg import norm

from rampedpyrox.core.thermogram import ModeledData

#define function to calculate phi
def _calc_phi(A, g, omega):
	'''
	Calculates phi, the discretized Lapalce transform f(Ea) result.
	Called by ``LaplaceTransform.calc_EC_inv()``.

	Args:
		A (np.ndarray): Laplace Transform matrix of shape [nT x nE].

		g (np.ndarray): Array of fraction of carbon remaining.

		omega (int): Omega value for Tikhonov Regularization.

	Returns:
		phi (np.ndarray): Array of the pdf of the distribution of Ea.

		resid_err (float): Residual RMSE between true and modeled thermogram.

		rgh_err (float): Roughness RMSE from Tikhonov Regularization.
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
	Calculates regularization matrix (R) for a given size, n.
	Called by ``_calc_phi()``.

	Args:
		n (int): Size of the regularization matrix will be [n+1 x n].

	Returns:
		R (np.ndarray): Regularization matrix of size [n+1 x n].

	References:
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

#function to round to sig fig
def _round_to_sigfig(vec, sig_figs=6):
	'''
	Rounds inputted vector to specified sig fig.
	Called by ``calc_L_curve``.

	Args:
		vec (np.ndarray): Array of data to round.

		sig_figs (int): Number of sig figs to round to. Defaults to 6.

	Returns:
		vec_round (np.ndarray): Rounded array.

	References:
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

#define function to calculate A matrix for LaplaceTransform
def calc_A(t, Tau, eps, logk0):
	'''
	Instance method to calculate the Laplace transform (A) matrix.
	Called by ``LaplaceTransform.__init__()``.

	Args:
		t (np.ndarray): Array of timepoints.

		Tau (np.ndarray): Array of temperature points.

		eps (np.ndarray): Array of Ea values.

		logk0 (int, float, or lambda): Arrhenius pre-exponential factor,
			either a constant value or a lambda function of Ea.

	Returns:
		A (np.ndarray): Laplace Transform matrix of shape [nT x nE].

	Examples:
		Calculating an A matrix for given input data::

			#t and Tau from RealData object rd
			eps = np.arange(50,350) #Ea range to calculate over
			logk0 = 10 #pre-exponential (Arrhenius) factor

			A = rp.LaplaceTransform(rd.t, rd.Tau, eps, logk0)

	References:
		R.L Braun and A.K. Burnham (1987) Analysis of chemical reaction kinetics
			using a distribution of activation energies and simpler models.
			*Energy & Fuels*, **1**, 153-161.

		\B. Cramer et al. (1998) Modeling isotope fractionation during primary
			cracking of natural gas: A reaction kinetic approach. *Chemical
			Geology*, **149**, 235-250.

		D.C. Forney and D.H. Rothman (2012) Common structure in the
			heterogeneity of plant-matter decay. *Journal of the Royal Society
			Interface*, rsif.2012.0122.

		D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
			respiration rates from decay time series. *Biogeosciences*, **9**,
			3601-3612.
	'''

	#set constants
	nT = len(Tau)
	nE = len(eps)
	k0 = 10**logk0 #s-1
	R = 8.314/1000 #kJ/mol/K
	del_t = t[1]-t[0] #s
	del_eps = eps[1]-eps[0] #kJ

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
	Instance method to calculate the L curve for a given matrix A and
	thermogram data g.
	Called by ``LaplaceTransform.calc_EC_inv()`` if omega is 'auto'.
	Called by ``LaplaceTransform.plot_L_curve()``.

	Args:
		A (np.ndarray): Laplace Transform matrix of shape [nT x nE].

		g (np.ndarray): Array of fraction of carbon remaining.

		log_om_min (int): Log10 of minimum omega value to search.
			Defaults to -3.

		log_om_max (int): Log10 of maximum omega value to search.
			Defaults to 2.

		nOm (int): Number of omega values to consider. Defaults to 100.

	Returns:
		om_best (float): Omega value that minimizes resididual RMSE and
			roughness RMSE.

		resid_vec (np.ndarray): Array of log10 of the residual RMSE.
		
		rgh_vec (np.ndarray): Array of the log10 of the roughness RMSE.
		
		omega_vec (np.ndarray): Array of the omega values considered.

	Examples:
		Basic implementation::

			#assuming A calculated as above and g from RealData object rd.
			om_best,resid_vec,rgh_vec,omega_vec = rp.calc_L_curve(A,g)

	References:
		D.C. Forney and D.H. Rothman (2012) Inverse method for calculating
			respiration rates from decay time series. *Biogeosciences*, **9**,
			3601-3612.

		P.C. Hansen (1987) Rank-deficient and discrete ill-posed problems:
			Numerical aspects of linear inversion (monographs on mathematical
			modeling and computation). *Society for Industrial and Applied
			Mathematics*.
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


class LaplaceTransform(object):
	'''
	Class for storing the A matrix and calculating forward/inverse results.

	Args:
		t (np.ndarray): Array of timepoints.

		Tau (np.ndarray): Array of temperature points.

		eps (np.ndarray): Array of Ea values.

		logk0 (int, float, or lambda): Arrhenius pre-exponential factor,
			either a constant value or a lambda function of Ea.

	Returns:
		lt (rp.LaplaceTransform): ``LaplaceTransform`` object.

	Raises:
		ValueError: If logk0 is not scalar, lambda function, or array of length nE.

	Examples:
		Calculating the Laplace Transform object and plotting the L-curve::

			#load modules
			import numpy as np
			import matplotlib.pyplot as plt

			eps = np.arange(50,350) #Ea range to calculate over
			logk0 = 10 #pre-exponential (Arrhenius) factor
			lt = rp.LaplaceTransform(rd.t,rd.Tau,eps,logk0)
			omega,ax = lt.plot_L_curve(rd)
	
	References:

		R.L Braun and A.K. Burnham (1987) Analysis of chemical reaction kinetics
			using a distribution of activation energies and simpler models.
			*Energy & Fuels*, **1**, 153-161.

		\B. Cramer et al. (1998) Modeling isotope fractionation during primary
			cracking of natural gas: A reaction kinetic approach. *Chemical
			Geology*, **149**, 235-250.

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
		Calculates the Energy Complex for a given Thermogram (inverse model).

		Args:
			tg (rp.RealData): ``RealData`` object containing the data to use
				for the inverse model.

			omega (int or str): Omega value for Tikhonov regularization, either
				an integer or 'auto'. Defaults to 'auto'.

		Returns:
			phi (np.ndarray): Array of the pdf of the distribution of Ea.

			resid_err (float): Residual RMSE between true and modeled thermogram.

			rgh_err (float): Roughness RMSE from Tikhonov Regularization.

			om_best (float): Best-fit omega value (assuming inputted omega='auto').

		Raises:
			ValueError: If the size of the A matrix does not match that of the 
				data in tg.
			ValueError: If inputted omega is not int, float, or 'auto'.

		Examples:
			Basic implementation::

				#assuming LaplaceTransform object lt and RealData object rd
				phi,resid_err,rgh_err,omega = lt.calc_EC_inv(rd,omega='auto')
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

		Args:
			ec (rp.EnergyComplex): ``EnergyComplex`` object containing the 
				invsersion model results.

		Returns:
			md (rp.ModeledData): ``ModeledData`` object containing the estimated
				thermogram using the inversion model results.

		Examples:
			Basic implementation::

				#assuming LaplaceTransform object lt and EnergyComplex object ec
				rd = lt.calc_TG_fwd(ec)
		'''

		#calculate the estimated g
		g_hat = np.inner(self.A,ec.phi_hat) #fraction

		#calculate g for each peak
		gp = np.inner(self.A,ec.peaks.T) #fractions
		
		#pass into a ModeledData object and return
		return ModeledData(self.t, self.Tau, g_hat, gp)


	def plot_L_curve(self, tg, ax=None, log_om_min=-3, log_om_max=2, nOm=100):
		'''
		Calculates the L curve w.r.t. a given thermogram and plots.

		Args:
			tg (rp.RealData): ``RealData`` object containing the data to use
				for the inverse model.

			ax (None or matplotlib.axis): Axis to plot on. If None, 
				creates an axis object to return. Defaults to None.

			log_om_min (int): Log10 of minimum omega value to search.
				Defaults to -3.

			log_om_max (int): Log10 of maximum omega value to search.
				Defaults to 2.

			nOm (int): Number of omega values to consider. Defaults to 100.

		Returns:
			om_best (float): Best-fit omega value.

			ax (matplotlib.axis): Updated axis with plotted data.

		Examples:
			Basic implemenation::
				
				#assuming LaplaceTransform object lt and RealData object rd
				omega,ax = lt.plot_L_curve(rd)
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

	def summary():
		'''
		Prints a summary of the LaplaceTransform object.
		'''
