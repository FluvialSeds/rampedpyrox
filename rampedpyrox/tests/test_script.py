#script to run thermogram deconvolution
import rampedpyrox as rp
import numpy as np


rd = rp.RealData('test_data/VBC10.csv')
eps = np.arange(50,350)
lt = rp.LaplaceTransform(rd.t,rd.Tau,eps,10)
phi,resid_err,rgh_err,omega = lt.calc_EC_inv(rd,omega=1)
ec = rp.EnergyComplex(eps,phi,nPeaks=5)
md = lt.calc_TG_fwd(ec)