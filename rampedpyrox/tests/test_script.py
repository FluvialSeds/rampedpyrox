# script to run thermogram deconvolution
import rampedpyrox as rp
import numpy as np
import os
import matplotlib.pyplot as plt


#function to load files
def gen_str(name):
	p = os.path.join(os.path.dirname(__file__), name)
	return p

data = gen_str('test_rpo_thermogram.csv')
sum_data = gen_str('test_rpo_isotopes.csv')

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
	combined=None,
	nPeaks='auto',
	omega=3,
	peak_shape='Gaussian',
	thres=0.02)

tg.forward_model(daem, ec)

ri = rp.RpoIsotopes.from_csv(
	sum_data,
	blk_corr = True,
	mass_err = 0.01)

ri.fit(
	daem, 
	ec, 
	tg,
	DEa=None,
	nIter=10)

# #fit different DEa values
# ri2 = rp.RpoIsotopes.from_csv(
# 	sum_data,
# 	blk_corr = True,
# 	mass_err = 0.01)

# ri2.fit(
# 	daem, 
# 	ec, 
# 	tg,
# 	DEa=0.001,
# 	nIter=10)

# ri3 = rp.RpoIsotopes.from_csv(
# 	sum_data,
# 	blk_corr = True,
# 	mass_err = 0.01)

# ri3.fit(
# 	daem, 
# 	ec, 
# 	tg,
# 	DEa=0.01,
# 	nIter=10)

# ri4 = rp.RpoIsotopes.from_csv(
# 	sum_data,
# 	blk_corr = True,
# 	mass_err = 0.01)

# ri4.fit(
# 	daem, 
# 	ec, 
# 	tg,
# 	DEa=0.1,
# 	nIter=10)

# #make plot

# fig,ax = plt.subplots(1,1,figsize=(9.5,6))

# ax.plot(tg.t, ri.d13C_product,
# 	linewidth = 2,
# 	color = 'k',
# 	label = r'$\Delta Ea$ = 0 J/mol')

# ax.plot(tg.t, ri2.d13C_product,
# 	linewidth = 2,
# 	color = 'r',
# 	label = r'$\Delta Ea$ = 1 J/mol')

# ax.plot(tg.t, ri3.d13C_product,
# 	linewidth = 2,
# 	color = 'b',
# 	label = r'$\Delta Ea$ = 10 J/mol')

# ax.plot(tg.t, ri4.d13C_product,
# 	linewidth = 2,
# 	color = 'g',
# 	label = r'$\Delta Ea$ = 100 J/mol')

# ax.legend(loc='best',frameon=False)
# ax.set_ylim([-35, -25])