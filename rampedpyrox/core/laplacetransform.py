import matplotlib.pyplot as plt
import numpy as np

from scipy.optimize import nnls
from numpy.linalg import norm

#define function to calculate phi
def _calc_phi(A, g, omega):
	'''
	Calculates phi, the discretized Lapalce transform f(Ea) result.
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
	Calculates the Laplace transform (A) matrix.
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
	Calculates the L curve for a given A and g
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
	'''

	def __init__(self, t, Tau, eps, logk0):
		'''
		Initializes the LaplaceTransform object.
		'''

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
			omega,_,_,_ = calc_L_curve(self.A, g)
		elif not isinstance(omega,(int,float)):
			#raise error
			raise ValueError('omega must be int, float, or "auto"')

		phi, resid_err, rgh_err = _calc_phi(self.A, g, omega)

		return phi, resid_err, rgh_err, omega

	def calc_TG_fwd(self, ec):
		'''
		Calculates the Thermogram for a given Energy Complex (forward model).
		'''

	def plot_L_curve(self, tg, ax=None, log_om_min=-3, log_om_max=2, nOm=100):
		'''
		Calculates the L curve w.r.t. a given thermogram and plots
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

		return om_best

	def summary():
		'''
		Prints a summary of the LaplaceTransform object.
		'''
