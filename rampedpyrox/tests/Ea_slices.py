'''
Script for calculating the average activation energy over each time slice in
which Fm data are collected
'''

import rampedpyrox as rp
import numpy as np
import os
import matplotlib.pyplot as plt

from rampedpyrox.results.results_helper import _rpo_cont_ctf

#function to load files
def gen_str(name):
	p = os.path.join(os.path.dirname(__file__), name)
	return p

data = gen_str('test_data/test_rpo_thermogram.csv')
sum_data = gen_str('test_data/test_rpo_isotopes.csv')

# data = gen_str('test_data/TS4.csv')
# sum_data = gen_str('test_data/TS4_sum.csv')

#run through inversion model
tg = rp.RpoThermogram.from_csv(
	data,
	bl_subtract = True,
	nt=250,
	ppm_CO2_err=5,
	T_err=3)

daem = rp.Daem.from_timedata(
	tg,
	# log10k0 = lambda x: 0.02*x + 5,
	log10k0=10,
	Ea_max=400,
	Ea_min=50,
	nEa=400)

ec = rp.EnergyComplex.inverse_model(
	daem, 
	tg,
	# combined=[(1,2),(6,7)],
	combined=None,
	nPeaks='auto',
	omega='auto',
	peak_shape='Gaussian',
	thres=0.02)

tg.forward_model(daem, ec)

ri = rp.RpoIsotopes.from_csv(
	sum_data,
	blk_corr = True,
	mass_err = 0.01)

# ri.fit(
# 	daem, 
# 	ec, 
# 	tg,
# 	DEa=0.0018,
# 	nIter=10)


#Now calculate the average and std. of Ea evolved over each timestep in which
# Fm data exist (rather than assuming Gaussians)


_, ind_min, ind_max, ind_wgh = _rpo_cont_ctf(ri, tg)

nEa = ec.nEa
nF = ri.nFrac

#make an empty matrix to store results
Ea_diff = np.zeros([nF, nEa])
mus = np.zeros(nF)
sigmas = np.zeros(nF)

#loop through each time window and calculate Ea_diff distribution
for i in range(nF):

	imin = ind_min[i]
	imax = ind_max[i]

	#p(E,t) at time 0
	pt0 = ec.f*daem.A[imin,:]

	#p(E,t) at time final
	ptf = ec.f*daem.A[imax,:]

	#difference -- i.e. p(E) evolved of Dt
	Dpt = pt0 - ptf

	#scale difference so that integral is 1
	scalar = 1/np.sum(Dpt*np.gradient(ec.Ea))
	Dpt = Dpt*scalar

	#store in matrix
	Ea_diff[i, :] = Dpt

	#calculate the mean and stdev
	mu = np.sum(ec.Ea*Dpt*np.gradient(ec.Ea))
	var = np.sum((ec.Ea - mu)**2 * Dpt*np.gradient(ec.Ea))

	mus[i] = mu
	sigmas[i] = var**0.5

#make a plot
fig,ax = plt.subplots(1,1)

ax.errorbar(
	mus, 
	1000*(ri.Fm_frac - 1), 
	xerr = sigmas, 
	yerr = 1000*ri.Fm_frac_std,
	fmt = 'o')

# ax.scatter(ec.peak_info['mu (kJ/mol)'],ri.cmpt_info['Fm'])

ax.set_xlim([100,300])
ax.set_ylim([-1000, 100])

ax.set_xlabel('activation energy, E (kJ/mol)')
ax.set_ylabel(r'$\Delta ^{14}C$ (‰)')

#make a second plot
fig2, ax2 = plt.subplots(1,1)

ax2.plot(ec.Ea, Ea_diff.T)

ax2.set_xlim([100,300])
# ax2.set_ylim([0, 0.07])

ax2.set_xlabel('activation energy, E (kJ/mol)')
ax2.set_ylabel(r'$p(t_f - t_0, E)$')


plt.show()



