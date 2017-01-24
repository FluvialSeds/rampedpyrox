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






	def test_R13_d13C(self):
		#go from d13C to R13 and back
		x = np.array([-50, -25, 0, 25])
		R13 = _d13C_to_R13(x)
		d13C = _R13_to_d13C(R13)

		assert_almost_equal(x.all(), d13C.all(), places=6)

	def test_kie_d13C(self):

		#assert output shapes and datatypes with scalar DEa
		d13C_peak, d13C_err = _kie_d13C(
			0, 
			[50, 100, 150, 200, 225],
			model,
			ratedata,
			[-30, -28, -26, -24, -22])

		assert_is_instance(d13C_peak, np.ndarray)
		assert_is_instance(d13C_err, float)
		assert_equal(len(d13C_peak), 4)
		assert_equal(d13C_peak.dtype, 'float')

	def test_kie_d13C_MC(self):

		#assert that a length error is raised if DEa is not right length
		assert_raises(
			LengthError,
			_kie_d13C_MC,
			[1,2,3],
			[50, 100, 150, 200, 225],
			model,
			1,
			result,
			ratedata)

		#assert that a length error is raised if ind_wgh is not right length
		assert_raises(
			LengthError,
			_kie_d13C_MC,
			0,
			[50, 100, 150, 200, 225],
			model,
			1,
			result,
			ratedata)


		pk, pk_std, pk_rmse = _kie_d13C_MC(
			0,
			[50, 75, 85, 100, 110, 125, 150, 175, 199],
			model,
			1,
			result,
			ratedata)

		#assert output data are right shape and type
		assert_is_instance(pk, np.ndarray)
		assert_is_instance(pk_std, np.ndarray)

		assert_is_instance(pk_rmse, float)
		assert_equal(len(pk), 4)
		assert_equal(len(pk_std), 4)
		assert_equal(pk.dtype, 'float')
		assert_equal(pk_std.dtype, 'float')

	def test_nnls_MC(self):

		pk, pk_std, pk_rmse = _nnls_MC(
			np.zeros([9,4]),
			1,
			-20,
			0.1)

		#assert output data are right shape and type
		assert_is_instance(pk, np.ndarray)
		assert_is_instance(pk_std, np.ndarray)

		assert_is_instance(pk_rmse, float)
		assert_equal(len(pk), 4)
		assert_equal(len(pk_std), 4)
		assert_equal(pk.dtype, 'float')
		assert_equal(pk_std.dtype, 'float')

	def test_R13_CO2(self):

		R13_CO2 = _R13_CO2(
			0,
			model,
			[0.01, 0.01, 0.01, 0.01],
			ratedata)

		#assert output is the right shape and type
		assert_is_instance(R13_CO2, np.ndarray)
		assert_equal(R13_CO2.dtype, 'float')
		assert_equal(len(R13_CO2), timedata.nt)

		#assert data values
		assert_almost_equal(R13_CO2.min(), 0.01, places=3)
		assert_almost_equal(R13_CO2.max(), 0.01, places=3)
		# print(R13_CO2.max())
		# print(R13_CO2.min())

		#make another instance with no DEa but variable R13 and check
		R13_CO2 = _R13_CO2(
			0,
			model,
			[0.01, 0.01, 0.025, 0.025],
			ratedata)

		assert_almost_equal(R13_CO2.min(), 0.01, places=2)
		assert_almost_equal(R13_CO2.max(), 0.025, places=2)
		# print(R13_CO2.min())
		# print(R13_CO2.max())

	def test_rpo_cont_ctf(self):

		cont, ind_min, ind_max, ind = _rpo_cont_ctf(
			result,
			timedata,
			ctf = True)

		#assert types
		assert_is_instance(cont, np.ndarray)
		assert_is_instance(ind_min, np.ndarray)
		assert_is_instance(ind_max, np.ndarray)
		assert_is_instance(ind, np.ndarray)

		#assert data types
		assert_equal(cont.dtype, 'float')
		assert_equal(ind_min.dtype, 'int')
		assert_equal(ind_max.dtype, 'int')
		assert_equal(ind.dtype, 'int')

		#assert shapes
		assert_equal(cont.shape[0], 9)
		assert_equal(cont.shape[1], 4)
		assert_equal(ind_min.shape[0], 9)
		assert_equal(ind_max.shape[0], 9)
		assert_equal(ind.shape[0], 9)

		#assert ind_max > ind > ind_min
		assert_equal((ind_max-ind > 0).all(), True)
		assert_equal((ind-ind_min > 0).all(), True)


class test_result_creation:

	#test bare-bones creation of instance
	def test_rpo_Fm_init(self):
		t_frac = [[1000, 200], [200, 300], [300, 1000]]
		t_frac = np.array(t_frac)
		Fm_frac = [1.0, 0.5, 0.0]

		#create instance
		ri = rp.RpoIsotopes(t_frac = t_frac,
							Fm_frac = Fm_frac)

		assert (ri.Fm_frac == Fm_frac).all()

	def test_rpo_t_init(self):
		t_frac = [[1000, 200], [200, 300], [300, 1000]]
		t_frac = np.array(t_frac)
		Fm_frac = [1.0, 0.5, 0.0]

		#create instance
		ri = rp.RpoIsotopes(t_frac = t_frac,
							Fm_frac = Fm_frac)

		assert (ri.t_frac == t_frac).all()

	#test importing form a csv file

	def test_rpo_m_from_csv(self):
		data = gen_str('test_rpo_isotopes.csv')

		#create instance
		ri = rp.RpoIsotopes.from_csv(data,
									blk_corr = True,
									mass_err = 0.01)

		assert hasattr(ri, 'm_frac')
		
	def test_rpo_d13C_from_csv(self):
		data = gen_str('test_rpo_isotopes.csv')

		#create instance
		ri = rp.RpoIsotopes.from_csv(data,
									blk_corr = True,
									mass_err = 0.01)

		assert hasattr(ri, 'd13C_frac')

	def test_rpo_Fm_from_csv(self):
		data = gen_str('test_rpo_isotopes.csv')

		#create instance
		ri = rp.RpoIsotopes.from_csv(data,
									blk_corr = True,
									mass_err = 0.01)

		assert hasattr(ri, 'Fm_frac')

	def test_rpo_t_from_csv(self):
		data = gen_str('test_rpo_isotopes.csv')

		#create instance
		ri = rp.RpoIsotopes.from_csv(data,
									blk_corr = True,
									mass_err = 0.01)

		assert hasattr(ri, 't_frac')


if __name__ == '__main__':

	import nose

	nose.runmodule(
		argv = [__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
		exit=False)
