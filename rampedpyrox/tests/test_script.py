#script to run thermogram deconvolution

rd = rp.RealData('test_data/5C_min.csv')
eps = np.arange(50,350)
lt = rp.LaplaceTransform(rd.t,rd.Tau,eps,10)
phi,resid_err,rgh_err,omega = lt.calc_EC_inv(rd,omega=5)
ec = rp.EnergyComplex(eps,phi,nPeaks=6)
md = lt.calc_TG_fwd(ec)