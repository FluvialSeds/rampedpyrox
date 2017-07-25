'''
This module contains ratedata module tests.
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

from rampedpyrox.core.exceptions import(
	ArrayError,
	LengthError,
	ScalarError,
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


class test_ratedata_creation:

	def test_bare_bones_input_types(self):

		#assert that manually importing data takes proper types
		#inputting string for E
		assert_raises(
			ArrayError, 
			rp.EnergyComplex,
			'[1,2,3]')

		#inputting negative values for E
		assert_raises(
			ArrayError, 
			rp.EnergyComplex,
			[-1, 0, 1, 2])

		#inputting string for f
		assert_raises(
			ArrayError, 
			rp.EnergyComplex,
			[1,2,3],
			p = '[1,2,3]')

	def test_bare_bones_creation_types(self):

		ec = rp.EnergyComplex([1,2,3], p = [1,2,3])

		#asserting types are floats
		assert_equal(ec.E.dtype, 'float')
		assert_equal(ec.p.dtype, 'float')

		#assert everything is an ndarray
		assert_is_instance(ec.E, np.ndarray)
		assert_is_instance(ec.p, np.ndarray)

	def test_inverse_model_creation(self):

		# # can't test warnings since no other model and timedata types
		# # currently exist.

		# #assert that non-Daems give warning
		# assert_warns(
		# 	UserWarning, 
		# 	rp.EnergyComplex.inverse_model,
		# 	daem,
		# 	non_thermogram,
		# 	omega=3)

		# #assert that non-RpoThermograms give warning
		# assert_warns(
		# 	UserWarning, 
		# 	rp.EnergyComplex.inverse_model,
		# 	non_daem,
		# 	thermogram,
		# 	omega=3)

		#assert that input types must be right
		assert_raises(
			ScalarError,
			rp.EnergyComplex.inverse_model,
			model,
			timedata,
			omega = [1,2,3])

	def test_input_estimated_instances(self):

		#create an inverse model to input estimated data
		ec = rp.EnergyComplex.inverse_model(
			model, 
			timedata,
			omega = 3)

		#make sure everything is the right type
		assert_is_instance(ec.resid, float)
		assert_is_instance(ec.rgh, float)

if __name__ == '__main__':

	import nose

	nose.runmodule(
		argv = [__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
		exit=False)

