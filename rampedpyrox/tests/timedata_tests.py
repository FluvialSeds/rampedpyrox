# timedata module tests

import numpy as np
import os
import pandas as pd

import rampedpyrox as rp

from nose.tools import(
	assert_almost_equal,
	assert_equal,
	assert_is_instance,
	assert_raises,
	)

from rampedpyrox.timedata.timedata_helper import(
	_rpo_extract_tg)

from rampedpyrox.core.exceptions import(
	# rpException,
	ArrayError,
	FileError,
	# FitError,
	LengthError,
	# RunModelError,
	# ScalarError,
	)

#function to load files
def gen_str(name):
	p = os.path.join(os.path.dirname(__file__), name)
	return p

file_str = gen_str('test_rpo_thermogram.csv')
file = pd.DataFrame.from_csv(file_str)

#test the timedata helper functions
class test_rpo_extract_tg:

	#import type tests
	def test_no_list(self):
		#assert that it doesn't take a list for `file`
		assert_raises(
			FileError, 
			_rpo_extract_tg,
			[1,2,3], 
			250,
			0)

	def test_no_garbage_string(self):
		#assert that it doesn't take a garbage string for `file`
		assert_raises(
			OSError, 
			_rpo_extract_tg,
			'garbage string', 
			250,
			0)

	def test_no_nt_array(self):
		#assert that it doesn't take an array for `nt`

		assert_raises(
			TypeError, 
			_rpo_extract_tg,
			file, 
			[1,2,3],
			0)

	def test_no_nt_string(self):
		#assert that it doesn't take a string for `nt`

		assert_raises(
			TypeError, 
			_rpo_extract_tg,
			file, 
			'250',
			0)

	def test_no_err_array(self):
		#assert that it doesn't take an array for `err`

		assert_raises(
			ValueError, 
			_rpo_extract_tg,
			file, 
			250,
			[1,2,3])

	def test_no_err_string(self):
		#assert that it doesn't take a string for `err`

		assert_raises(
			TypeError, 
			_rpo_extract_tg,
			file, 
			250,
			'0')

	#data tests
	def test_g_range(self):
		#assert that g ranges from 0 to 1

		g, g_std, t, T = _rpo_extract_tg(
			file,
			250,
			1)

		assert_almost_equal(np.max(g), 1, places=3)
		assert_almost_equal(np.min(g), 0, places=3)

	def test_array_lengths(self):
		#assert that all arrays are length nt

		g, g_std, t, T = _rpo_extract_tg(
			file,
			250,
			1)

		assert_equal(len(g), 250)
		assert_equal(len(g_std), 250)
		assert_equal(len(t), 250)
		assert_equal(len(T), 250)

	def test_array_types(self):
		#assert that all arrays are np.ndarray

		g, g_std, t, T = _rpo_extract_tg(
			file,
			250,
			1)

		assert_is_instance(g, np.ndarray)
		assert_is_instance(g_std, np.ndarray)
		assert_is_instance(t, np.ndarray)
		assert_is_instance(T, np.ndarray)

class test_thermogram_creation:

	def test_bare_bones_input_types(self):
		#assert that manually importing data takes proper types
		#inputting string for t
		assert_raises(
			ArrayError, 
			rp.RpoThermogram,
			'[1,2,3]', 
			[1,2,3])

		#inputting string for T
		assert_raises(
			ArrayError, 
			rp.RpoThermogram,
			[1,2,3], 
			'[1,2,3]')

		#inputting different length arrays
		assert_raises(
			LengthError, 
			rp.RpoThermogram,
			[1,2,3], 
			[1,2,3,4])

		#inputting different length g array
		assert_raises(
			LengthError, 
			rp.RpoThermogram,
			[1,2,3], 
			[1,2,3],
			[0,0.1,0.2,0.3])

		#inputting g beyond (0-1)
		assert_raises(
			ArrayError, 
			rp.RpoThermogram,
			[1,2,3], 
			[1,2,3],
			[0,1, 2])

	def test_bare_bones_creation_types(self):
		#assert that data are stored properly
		tg = rp.RpoThermogram(
			[1,2,3], 
			[1,2,3], 
			g = [0,0.1,0.2],
			g_std = [1,2,3],
			T_std = [1,2,3])

		assert_is_instance(tg.g, np.ndarray)
		assert_is_instance(tg.g_std, np.ndarray)
		assert_is_instance(tg.t, np.ndarray)
		assert_is_instance(tg.T, np.ndarray)
		assert_is_instance(tg.T_std, np.ndarray)






