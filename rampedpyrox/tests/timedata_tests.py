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

from rampedpyrox.timedata.timedata_helper import(
	_rpo_extract_tg)

from rampedpyrox.core.exceptions import(
	# rpException,
	ArrayError,
	FileError,
	# FitError,
	LengthError,
	RunModelError,
	# ScalarError,
	StringError,
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
	def test_input_types(self):
		#assert that it doesn't take a list for `file`
		assert_raises(
			FileError, 
			_rpo_extract_tg,
			[1,2,3], 
			250,
			0)

		#assert that it doesn't take a garbage string for `file`
		assert_raises(
			OSError, 
			_rpo_extract_tg,
			'garbage string', 
			250,
			0)

		#assert that it doesn't take an array for `nt`
		assert_raises(
			TypeError, 
			_rpo_extract_tg,
			file, 
			[1,2,3],
			0)

		#assert that it doesn't take a string for `nt`
		assert_raises(
			TypeError, 
			_rpo_extract_tg,
			file, 
			'250',
			0)

		#assert that it doesn't take an array for `err`
		assert_raises(
			ValueError, 
			_rpo_extract_tg,
			file, 
			250,
			[1,2,3])

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

#test creating thermogram instances
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

		#input isothermal data
		assert_warns(
			UserWarning,
			rp.RpoThermogram,
			[1,2,3],
			[1,1,1])

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

	def test_from_csv_creation(self):
		#assert that data from csv are stored
		tg = rp.RpoThermogram.from_csv(
			file,
			nt = 250,
			ppm_CO2_err = 5,
			T_err = 3)

		assert_is_instance(tg.g, np.ndarray)
		assert_is_instance(tg.g_std, np.ndarray)
		assert_is_instance(tg.t, np.ndarray)
		assert_is_instance(tg.T, np.ndarray)
		assert_is_instance(tg.T_std, np.ndarray)

	def test_from_data_frame_or_string(self):
		#assert that inputting the file as a df or string are equal		
		tg = rp.RpoThermogram.from_csv(
			file,
			nt = 250,
			ppm_CO2_err = 5,
			T_err = 3)

		tg_str = rp.RpoThermogram.from_csv(
			file_str,
			nt = 250,
			ppm_CO2_err = 5,
			T_err = 3)

		assert_equal(all(tg.g), all(tg_str.g))
		assert_equal(all(tg.g_std), all(tg_str.g_std))
		assert_equal(all(tg.t), all(tg_str.t))
		assert_equal(all(tg.T), all(tg_str.T))
		assert_equal(all(tg.T_std), all(tg_str.T_std))

#test inputting data into thermogram instancece
class test_thermogram_modeled_input:

	def test_input_warnings_and_raises(self):

		tg = rp.RpoThermogram.from_csv(
			file,
			nt = 250,
			ppm_CO2_err = 5,
			T_err = 3)

		ec = rp.EnergyComplex([1,2,3])
		daem = rp.Daem.from_timedata(tg) #different nEa
		daem2 = rp.Daem.from_ratedata(ec, nt = 300) #different nt
		daem3 = rp.Daem.from_ratedata(ec, nt = 250) #right shape


		# can't test warnings since no other model and ratedata types
		# currently exist.

		# #assert that non-Daems give warning
		# assert_warns(
		# 	UserWarning, 
		# 	tg.forward_model,
		# 	np.array([1,2,3]), #made-up array model
		# 	ec)

		# #assert that non-EnergyComplexes give warning
		# assert_warns(
		# 	UserWarning, 
		# 	tg.forward_model,
		# 	daem,
		# 	np.array([1,2,3])) #made-up array ratedata

		#assert that different length nEa raises exception
		assert_raises(
			ArrayError, 
			tg.forward_model,
			daem,
			ec)

		#assert that different length t raises exception 
		assert_raises(
			ArrayError, 
			tg.forward_model,
			daem2,
			ec)

		#assert that non-inverse-modeled EC raises esception
		assert_raises(
			RunModelError, 
			tg.forward_model,
			daem3,
			ec)

	def test_forward_model_instances(self):

		#assert that instance gets attributes
		#put it all together
		tg = rp.RpoThermogram.from_csv(
			file,
			nt = 250,
			ppm_CO2_err = 5,
			T_err = 3)

		daem = rp.Daem.from_timedata(tg)

		ec = rp.EnergyComplex.inverse_model(
			daem,
			tg,
			omega = 3)

		tg.forward_model(daem, ec)

		#assert instances
		assert_is_instance(tg.dgamdt, np.ndarray)
		assert_is_instance(tg.dgamdT, np.ndarray)
		assert_is_instance(tg.dcmptdt, np.ndarray)
		assert_is_instance(tg.dcmptdT, np.ndarray)
		assert_is_instance(tg.gam, np.ndarray)

		assert_is_instance(tg.dof, int)
		assert_is_instance(tg.nPeak, int)
		assert_is_instance(tg.red_chi_sq, float)
		assert_is_instance(tg.rmse, float)

	def test_forward_model_lengths(self):

		#assert that instance gets attributes
		#put it all together
		tg = rp.RpoThermogram.from_csv(
			file,
			nt = 250,
			ppm_CO2_err = 5,
			T_err = 3)

		daem = rp.Daem.from_timedata(tg)

		ec = rp.EnergyComplex.inverse_model(
			daem,
			tg,
			omega = 3)

		tg.forward_model(daem, ec)

		#assert instances
		assert_equal(len(tg.dgamdt), 250)
		assert_equal(len(tg.dgamdT), 250)
		assert_equal(len(tg.dcmptdt), 250)
		assert_equal(len(tg.dcmptdT), 250)
		assert_equal(len(tg.gam), 250)

#test plotting data
class test_thermogram_plots:

	def test_axis_strings(self):
	#assert that strings must be right

		tg = rp.RpoThermogram.from_csv(
			file,
			nt = 250,
			ppm_CO2_err = 5,
			T_err = 3)

		assert_raises(
			StringError,
			tg.plot,
			xaxis = 'garbage',
			yaxis = 'fraction')

		assert_raises(
			StringError,
			tg.plot,
			xaxis = 'time',
			yaxis = 'garbage')

if __name__ == '__main__':

	import nose

	nose.runmodule(
		argv = [__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
		exit=False)

