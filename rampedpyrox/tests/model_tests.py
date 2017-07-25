# *TODO: add L-curve testing!

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
	_calc_ghat,
	_calc_p,
	_calc_R,
	_rpo_calc_A)

from rampedpyrox.core.exceptions import(
	ArrayError,
	LengthError,
	StringError,
	)

#function to load files
def gen_str(name):
	p = os.path.join(os.path.dirname(__file__), name)
	return p

file_str = gen_str('test_data/thermogram.csv')
file = pd.DataFrame.from_csv(file_str)

#create some timedata, model, and ratedata instances
timedata = rp.RpoThermogram.from_csv(
	file_str,
	nt = 250,
	bl_subtract = True)

model = rp.Daem.from_timedata(
	timedata,
	nE = 300)

ratedata = rp.EnergyComplex.inverse_model(
	model, 
	timedata,
	lam = 3)

#test the model helper functions
class test_model_helper_functions:

	def test_calc_ghat(self):
		ghat = _calc_ghat(model, ratedata)

		#assert that the shape is right
		assert_equal(ghat.shape[0], 250)

		#check that the type is right
		assert_is_instance(ghat, np.ndarray)
		assert_equal(ghat.dtype, 'float')

	def test_calc_p(self):
		p, resid, rgh = _calc_p(model, timedata, 3)

		#assert that the length is right
		assert_equal(p.shape[0], 300)

		#assert that the type is right
		assert_is_instance(p, np.ndarray)
		assert_equal(p.dtype, float)
		assert_is_instance(resid, float)
		assert_is_instance(rgh, float)

		#assert that the integral of f is one
		P = p*np.gradient(ratedata.E)
		assert_almost_equal(np.sum(P), 1, places=3)

		#assert that f is nonnegative
		assert_almost_equal(np.min(P), 0, places=3)

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
			ratedata.E, 
			10, 
			timedata.t, 
			timedata.T)

		assert_equal(A.shape[0], 250)
		assert_equal(A.shape[1], 300)
		assert_is_instance(A, np.ndarray)
		assert_equal(A.dtype, float)

		#assert that the range is 0 - 1 (divide by dE)
		a = np.divide(A, np.gradient(ratedata.E))
		assert_equal(np.max(a), 1)
		assert_equal(np.min(a), 0)

	def test_A_takes_lambda(self):
		#input lambda and make sure it works
		log10omega = lambda ea: 0.02*ea + 5

		A = _rpo_calc_A(
			ratedata.E, 
			log10omega, 
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
			ratedata.E,
			'10',
			timedata.t,
			timedata.T)

		#inputting string for t
		assert_raises(
			ArrayError, 
			rp.Daem,
			ratedata.E,
			10,
			'[1,2,3]',
			timedata.T)

		#inputting string for t
		assert_raises(
			ArrayError, 
			rp.Daem,
			ratedata.E,
			10,
			timedata.t,
			'[1,2,3]')

		#inputting different length t and T
		assert_raises(
			LengthError, 
			rp.Daem,
			ratedata.E,
			10,
			[1,2,3],
			[1,2,3,4])

		#input isothermal data
		assert_warns(
			UserWarning,
			rp.Daem,
			ratedata.E,
			10,
			[1,2,3],
			[1,1,1])

	# def test_from_data_warnings_and_raises(self):

	# 	#can't test warnings since no other model and ratedata types
	# 	# currently exist.

	# 	#assert that non-thermograms give warning
	# 	assert_warns(
	# 		UserWarning, 
	# 		rp.Daem.from_timedata,
	# 		timedata) #non-Thermogram timedata instance

	# 	#assert that non-ECs give warning
	# 	assert_warns(
	# 		UserWarning, 
	# 		rp.Daem.from_ratedata,
	# 		ratedata) #non-Thermogram timedata instance

if __name__ == '__main__':

	import nose

	nose.runmodule(
		argv = [__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
		exit=False)
