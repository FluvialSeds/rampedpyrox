#script to run thermogram deconvolution
import rampedpyrox as rp
import numpy as np
import matplotlib.pyplot as plt

#plt.close('all')

# rd = rp.RealData('test_data/5C_min_true.csv',nT=250)
# eps = np.arange(50,350,.5)
# lt = rp.LaplaceTransform(rd.t,rd.Tau,eps,10)
# phi,resid_err,rgh_err,omega = lt.calc_EC_inv(rd,omega=1)
# ec = rp.EnergyComplex(eps,phi,
# 	nPeaks='auto',
# 	thres=0.02,
# 	combine_last=None,
# 	DEa=0.0018)
# md = lt.calc_TG_fwd(ec)


# # ir = rp.IsotopeResult('test_data/5C_min_sum.csv',lt, ec, 
# #  	blk_corr=True,
# #  	mass_rsd=0.01,
# #  	add_noise=True)

# # ec.plot()
# # ax = md.plot()
# # rd.plot(ax=ax)

# # plt.show()

data = 'test_data/5C_min_true.csv'
sum_data = 'test_data/5C_min_sum.csv'

tg = rp.RpoThermogram.from_csv(data, nt = 250)
daem = rp.Daem.from_timedata(tg, nEa = 300)
ec = rp.EnergyComplex.inverse_model(daem, tg,
	omega = 3,
	thres = 0.02,
	combined = [(6,7)])
tg.forward_model(daem, ec)
ri = rp.RpoIsotopes.from_csv(sum_data)
ri.fit(daem, ec, tg, DEa = 0.0018)