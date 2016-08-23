#script to run thermogram deconvolution
import rampedpyrox as rp
import numpy as np
import matplotlib.pyplot as plt

#plt.close('all')

rd = rp.RealData('test_data/TS1.csv')
eps = np.arange(50,350)
lt = rp.LaplaceTransform(rd.t,rd.Tau,eps,10)
phi,resid_err,rgh_err,omega = lt.calc_EC_inv(rd,omega=1)
ec = rp.EnergyComplex(eps,phi,nPeaks='auto',thres=0.02)
md = lt.calc_TG_fwd(ec)

ec.plot()
ax = md.plot()
rd.plot(ax=ax)

plt.show()