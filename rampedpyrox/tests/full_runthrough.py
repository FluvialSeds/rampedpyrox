# script to run thermogram deconvolution
import rampedpyrox as rp
import numpy as np
import os
import matplotlib.pyplot as plt


#function to load files
def gen_str(name):
	p = os.path.join(os.path.dirname(__file__), name)
	return p

#paths to data files
data = gen_str('test_data/thermogram.csv')
# sum_data = gen_str('test_data/res_no_iso.csv') # FOR TESTING W/OUT ISOTOPES
sum_data = gen_str('test_data/isotopes.csv')

#calculate thermogram
tg = rp.RpoThermogram.from_csv(
	data,
	bl_subtract = True,
	nt = 250)

#calculate DAEM
daem = rp.Daem.from_timedata(
	tg,
	log10omega = 10, #value advocated in Hemingway (2017) Biogeosciences
	E_max = 400, #can change if too high
	E_min = 50, #can change if too low
	nE = 400)

#calculate energy complex
ec = rp.EnergyComplex.inverse_model(
	daem, 
	tg,
	lam = 1) #can replace with best-fit value if known

#forward model estimated thermogram back onto tg
tg.forward_model(daem, ec)

#calculate isotope results
ri = rp.RpoIsotopes.from_csv(
	sum_data,
	daem,
	ec,
	blk_corr = True,
	bulk_d13C_true = None)

#plot results
# fig, ax = plt.subplots(1,3)
# ax[0] = ri.plot(ax = ax[0], plt_var = 'p0E')
# ax[1] = ri.plot(ax = ax[1], plt_var = 'Fm', plt_corr = True)
# ax[2] = ri.plot(ax = ax[2], plt_var = 'd13C', plt_corr = True)


plt.show()
