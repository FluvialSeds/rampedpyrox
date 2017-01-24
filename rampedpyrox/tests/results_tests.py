'''
This module contains result module tests.
'''

import numpy as np
import os
import pandas as pd

import rampedpyrox as rp

from nose.tools import(
	assert_almost_equal,
	assert_equal,
	assert_is_instance,
	assert_not_equal,
	assert_raises,
	assert_true,
	# assert_warns,
	)

from rampedpyrox.results.results_helper import(
	_calc_cutoff,
	_calc_E_frac,
	_rpo_blk_corr,
	_rpo_extract_iso,
	_rpo_kie_corr,
	_rpo_mass_bal_corr,
	)

from rampedpyrox.core.exceptions import(
	# rpException,
	# ArrayError,
	# FileError,
	# FitError,
	LengthError,
	# RunModelError,
	# ScalarError,
	# StringError,
	)

#function to load files
def gen_str(name):
	p = os.path.join(os.path.dirname(__file__), name)
	return p

file_str = gen_str('test_data/thermogram.csv')
file = pd.DataFrame.from_csv(file_str)

res_str = gen_str('test_data/isotopes.csv')
res = pd.DataFrame.from_csv(res_str)


#create some timedata, model, and ratedata instances
timedata = rp.RpoThermogram.from_csv(
	file_str,
	nt = 250)

model = rp.Daem.from_timedata(
	timedata,
	nE = 300)

ratedata = rp.EnergyComplex.inverse_model(
	model, 
	timedata,
	omega = 3)

timedata.forward_model(model, ratedata)

result = rp.RpoIsotopes.from_csv(
	res_str,
	model,
	ratedata,
	blk_corr = False,
	mass_err = 0.01,
	bulk_d13C_true = None)


class test_results_helper_functions:

	def test_calc_cutoff(self):
		ind_min, ind_max = _calc_cutoff(result, model)

		#assert lengths and types
		assert_is_instance(ind_min, np.ndarray)
		assert_is_instance(ind_max, np.ndarray)
		assert_equal(len(ind_min), len(ind_max))
		assert_equal(ind_max.dtype, 'int')
		assert_equal(ind_min.dtype, 'int')

		#assert ind_max is always greater than ind_min
		assert_true(np.all(result.t_frac[1,:] > result.t_frac[0,:]))

	def test_calc_E_frac(self):
		E_frac, E_frac_std, p_frac = _calc_E_frac(
			result, 
			model, 
			ratedata)

		#test shapes and dtypes
		assert_equal(len(E_frac), len(E_frac_std))
		assert_equal(len(E_frac), np.shape(p_frac)[0])
		assert_equal(len(ratedata.E), np.shape(p_frac)[1])
		assert_equal(E_frac.dtype, 'float')
		assert_equal(E_frac_std.dtype, 'float')
		assert_equal(p_frac.dtype, 'float')

		#test that p_frac sums to p
		phat = np.sum(p_frac,axis=0)
		assert_almost_equal(all(ratedata.p),all(phat),places=3)

		#assert phat integrates to 1.0
		pint = np.sum(phat*np.gradient(ratedata.E))
		assert_almost_equal(pint, 1.0, places = 3)


class test_result_creation:

	#test bare-bones creation of instance
	def test_rpo_init(self):
		t_frac = [[100, 200], [200, 300], [300, 1000]]
		t_frac = np.array(t_frac)
		Fm_raw = [1.0, 0.5, 0.0]

		#create instance
		ri = rp.RpoIsotopes(
			model,
			ratedata,
			t_frac = t_frac,
			Fm_raw = Fm_raw)

		fm = all(ri.Fm_raw - Fm_raw == 0)
		# t = all(ri.t_frac - t_frac == 0)

		assert_true(fm)
		# assert_true(t)

	#test importing form a csv file
	def test_rpo_from_csv(self):

		#create instance
		ri = rp.RpoIsotopes.from_csv(
			res_str,
			model,
			ratedata,
			blk_corr = True,
			mass_err = 0.01,
			DE = None)

		#extract all expected attributes
		d13C_raw = getattr(ri, 'd13C_raw', None)
		d13C_raw_std = getattr(ri, 'd13C_raw_std', None)
		Fm_raw = getattr(ri, 'Fm_raw', None)
		Fm_raw_std = getattr(ri, 'Fm_raw_std', None)
		m_raw = getattr(ri, 'm_raw', None)
		m_raw_std = getattr(ri, 'm_raw_std', None)
		d13C_corr = getattr(ri, 'd13C_corr', None)
		d13C_corr_std = getattr(ri, 'd13C_corr_std', None)
		Fm_corr = getattr(ri, 'Fm_corr', None)
		Fm_corr_std = getattr(ri, 'Fm_corr_std', None)
		m_corr = getattr(ri, 'm_corr', None)
		m_corr_std = getattr(ri, 'm_corr_std', None)

		t_frac = getattr(ri, 't_frac', None)
		E_frac = getattr(ri, 'E_frac', None)
		E_frac_std = getattr(ri, 'E_frac_std', None)

		#assert that they all exist
		assert_not_equal(d13C_raw, None)
		assert_not_equal(d13C_raw_std, None)
		assert_not_equal(d13C_corr, None)
		assert_not_equal(d13C_corr_std, None)

		assert_not_equal(Fm_raw, None)
		assert_not_equal(Fm_raw_std, None)
		assert_not_equal(Fm_corr, None)
		assert_not_equal(Fm_corr_std, None)

		assert_not_equal(m_raw, None)
		assert_not_equal(m_raw_std, None)
		assert_not_equal(m_corr, None)
		assert_not_equal(m_corr_std, None)

		assert_not_equal(t_frac, None)
		assert_not_equal(E_frac, None)
		assert_not_equal(E_frac_std, None)


if __name__ == '__main__':

	import nose

	nose.runmodule(
		argv = [__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
		exit=False)
