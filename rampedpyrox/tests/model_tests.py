'''
This module contains timedata module tests,
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
	assert_warns,
	)

from rampedpyrox.model.model_helper import(
	_calc_cmpt,
	_calc_f,
	_calc_R,
	_rpo_calc_A)

from rampedpyrox.core.exceptions import(
	# rpException,
	ArrayError,
	# FileError,
	# FitError,
	LengthError,
	# RunModelError,
	# ScalarError,
	StringError,
	)

#function to load files
def gen_str(name):
	p = os.path.join(os.path.dirname(__file__), name)
	return p

file_str = gen_str('test_rpo_thermogram.csv')
file = pd.DataFrame.from_csv(file_str)

#create some timedata, model, and ratedata instances
timedata = rp.RpoThermogram.from_csv(
	file_str,
	nt = 250)

model = rp.Daem.from_timedata(
	timedata,
	nEa = 300)

ratedata = rp.EnergyComplex.inverse_model(
	model, 
	timedata,
	nPeaks = 3, 
	omega = 3)

#test the timedata helper functions
class test_model_helper_functions:

	def test_calc_cmpt(self):
		x = _calc_cmpt(model, ratedata)

		#assert that the shape is right
		assert_equal(x.shape[0], 250)
		assert_equal(x.shape[1], 3)

		#check that the type is right
		assert_is_instance(x, np.ndarray)
		assert_equal(x.dtype, 'float')

	def test_calc_f(self):
		f, resid_rmse, rgh_rmse = _calc_f(model, timedata, 3)

		#assert that the length is right
		assert_equal(f.shape[0], 300)

		#assert that the type is right
		assert_is_instance(f, np.ndarray)
		assert_equal(f.dtype, float)
		assert_is_instance(resid_rmse, float)
		assert_is_instance(rgh_rmse, float)

		#assert that the integral of f is one
		F = f*np.gradient(ratedata.Ea)
		assert_almost_equal(np.sum(F), 1, places=3)

		#assert that f is nonnegative
		assert_almost_equal(np.min(f), 0, places=3)

	def test_calc_R(self):
		#assert that R is the right shape and only contains -1, 0, 1
		R = _calc_R(300)

		#assert shape and type
		assert_equal(R.shape[0], 301)
		assert_equal(R.shape[1], 300)
		assert_is_instance(R, np.ndarray)
		assert_equal(R.dtype, 'float')

		#make sure the values are right
		assert_equal(np.max(R), 1)
		assert_equal(np.min(R), -1)

		assert_equal(R[0,0], 1)
		assert_equal(R[-1,-1], -1)

	def test_rpo_calc_A(self):
		#assert that A is the right shape and type
		A = _rpo_calc_A(
			ratedata.Ea, 
			10, 
			timedata.t, 
			timedata.T)

		assert_equal(A.shape[0], 250)
		assert_equal(A.shape[1], 300)
		assert_is_instance(A, np.ndarray)
		assert_equal(A.dtype, float)

		#assert that the range is 0 - 1 (divide by dEa)
		a = np.divide(A, np.gradient(ratedata.Ea))
		assert_equal(np.max(a), 1)
		assert_equal(np.min(a), 0)

	def test_A_takes_lambda(self):
		#input lambda and make sure it works
		log10k0 = lambda ea: 0.02*ea + 5

		A = _rpo_calc_A(
			ratedata.Ea, 
			10, 
			timedata.t, 
			timedata.T)

#test creating model instances
class test_model_creation:

	def test_bare_bones_input_types(self):
		#assert that manually importing data takes proper types
		#inputting string for Ea
		assert_raises(
			ArrayError, 
			rp.Daem,
			'5',
			10,
			timedata.t,
			timedata.T)

		#inputting string for log10k0
		assert_raises(
			ArrayError, 
			rp.Daem,
			ratedata.Ea,
			'10',
			timedata.t,
			timedata.T)

		#inputting string for t
		assert_raises(
			ArrayError, 
			rp.Daem,
			ratedata.Ea,
			10,
			'[1,2,3]',
			timedata.T)

		#inputting string for t
		assert_raises(
			ArrayError, 
			rp.Daem,
			ratedata.Ea,
			10,
			timedata.t,
			'[1,2,3]')

		#inputting different length t and T
		assert_raises(
			LengthError, 
			rp.Daem,
			ratedata.Ea,
			10,
			[1,2,3],
			[1,2,3,4])

		#input isothermal data
		assert_warns(
			UserWarning,
			rp.Daem,
			ratedata.Ea,
			10,
			[1,2,3],
			[1,1,1])








