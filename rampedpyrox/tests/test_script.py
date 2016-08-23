#script to run thermogram deconvolution
import rampedpyrox as rp
import numpy as np


rd = rp.RealData('test_data/TS1.csv')
eps = np.arange(50,350)
lt = rp.LaplaceTransform(rd.t,rd.Tau,eps,10)
phi,resid_err,rgh_err,omega = lt.calc_EC_inv(rd,omega='auto')
ec = rp.EnergyComplex(eps,phi,nPeaks='auto')
md = lt.calc_TG_fwd(ec)

ec.plot()
md.plot()