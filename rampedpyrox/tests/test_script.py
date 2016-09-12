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
	combined=[(1,2), (6, 7)],
	nPeaks='auto',
	omega=3,
	peak_shape='Gaussian',
	thres=0.02)

tg.forward_model(daem, ec)

ri = rp.RpoIsotopes.from_csv(
	sum_data,
	blk_corr=True,
	mass_err=0.01)

ri.fit(
	daem, 
	ec, 
	tg,
	DEa=0.001,
	nIter=10)




