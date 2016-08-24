#script to run thermogram deconvolution
import rampedpyrox as rp
import numpy as np
import matplotlib.pyplot as plt

#plt.close('all')

rd = rp.RealData('test_data/5C_min_true.csv')
eps = np.arange(50,350)
lt = rp.LaplaceTransform(rd.t,rd.Tau,eps,10)
phi,resid_err,rgh_err,omega = lt.calc_EC_inv(rd,omega=1)
ec = rp.EnergyComplex(eps,phi,nPeaks='auto',thres=0.03,combine_last=2)
md = lt.calc_TG_fwd(ec)


ir = rp.IsotopeResult('test_data/5C_min_sum.csv',md)

ec.plot()
ax = md.plot()
rd.plot(ax=ax)

plt.show()