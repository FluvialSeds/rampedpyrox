'''
This module contains a (sloppy) run-through generating fake data and checking 
model performance.
'''

import numpy as np
import os
import pandas as pd
import rampedpyrox as rp

from scipy.interpolate import interp1d

from rampedpyrox.ratedata.ratedata_helper import(
	_gaussian)

from rampedpyrox.results.results_helper import(
	_rpo_cont_ptf)


#INVERSE MODEL FAKE DATA
nt = 350
tmax = 10000
nEa = 300
Ea_min = 50
Ea_max = 350
nInterp = 10000
nIter = 10000

noise_pct = 0.01 #fractional ppmCO2 noise
T_noise = 0.1 #C, 0.1 roughly equal to +- 0.2 C/min dTdt
d13C_noise = 0.2
Fm_noise = 0.02


beta = 5. #C/min
logk = 10.

#make 3-Gaussian Ea distribution
Ea = np.linspace(Ea_min, Ea_max, nEa)
mu = [150, 200, 250]
sigma = [5, 10, 15]
rel_area = [.25, .25, .5]
d13C = [-20, -10, 0]
Fm = [1.0, 0.5, 0.0]

y = _gaussian(Ea, mu, sigma)
f = np.sum(y*rel_area, axis = 1)

#forward-model onto thermogram
tds = np.linspace(0, tmax, nt)
Tds = 373 + (beta/60)*tds

daem = rp.Daem(Ea, logk, tds, Tds)
gds = np.inner(daem.A, f)

#interpolate to make nInterp points
t = np.linspace(0, tmax, nInterp)
T = 373 + (beta/60)*t

func = interp1d(tds, gds)
g = func(t)

#calculate ppmCO2 at each timepoint and add noise
ppmCO2 = -np.gradient(g)/np.gradient(t)

ppmCO2 = ppmCO2 + np.random.randn(nInterp)*noise_pct*ppmCO2.max()
T = T + np.random.randn(nInterp)*T_noise

#make a dataframe
time = pd.to_datetime(t, unit='s')


#multiply ppmCO2 by 1e6 to get in reasonable ppm value range
#subtract 273 from T to get back to C
df = pd.DataFrame(
	np.column_stack((T-273,ppmCO2*1e6)),
	index = time,
	columns = ['temp','CO2_scaled'])

################################
# NOISY INSTANCES TO BE TESTED #
################################

#make timedata, model, and ratedata instances
tg = rp.RpoThermogram.from_csv(
	df,
	nt = nt)

daem = rp.Daem.from_timedata(tg, 
	log10k0 = logk, 
	Ea_max = Ea_max, 
	Ea_min = Ea_min, 
	nEa = nEa)

ec = rp.EnergyComplex.inverse_model(
	daem,
	tg,
	combined = None,
	nPeaks = 3,
	omega = 3,
	peak_shape = 'Gaussian',
	thres = 0.05)

tg.forward_model(daem, ec)

##################
# TRUE INSTANCES #
##################

ec_true = rp.EnergyComplex(Ea, f=f)
ec_true.input_estimated(y*rel_area)

tg_true = rp.RpoThermogram(tds, Tds)
tg_true.forward_model(daem, ec_true)

#calculate fraction contributions, masses, and isotopes
d13C_peak = [-30., -20., 0.]
Fm_peak = [1.0, 0.5, 0.0]

#say 5 fractions
t_frac = [[1500, 3000],
		[3000, 4500],
		[4500, 6000],
		[6000, 8000],
		[8000, 10000]]

t_frac = np.array([[1500, 3000],
					[3000, 4500],
					[4500, 6000],
					[6000, 8000],
					[8000, 10000]])

ri = rp.RpoIsotopes(t_frac = t_frac)

#calculate peak to fraction contributions
cont_ptf, _, _, _ = _rpo_cont_ptf(ri, tg_true, ptf=True)
cont_ftp, _, _, _ = _rpo_cont_ptf(ri, tg_true, ptf=False)


# d13C_peak = np.outer(np.ones(5),d13C_peak)
# Fm_peak = np.outer(np.ones(5),Fm_peak)

d13C_frac = np.inner(cont_ptf, d13C_peak)
Fm_frac = np.inner(cont_ptf, Fm_peak)
m_frac = np.inner(cont_ftp, rel_area)*100

#add some noise
# d13C_frac = d13C_frac + np.random.randn(5)*d13C_noise
# Fm_frac = Fm_frac + np.random.randn(5)*Fm_noise

ri2 = rp.RpoIsotopes(
	d13C_frac = d13C_frac,
	d13C_frac_std = d13C_noise,
	Fm_frac = Fm_frac,
	Fm_frac_std = Fm_noise,
	m_frac = m_frac,
	m_frac_std = 0.01*m_frac,
	t_frac = t_frac)

#fit results
ri2.fit(daem,
	ec,
	tg,
	DEa = None,
	nIter = nIter)


