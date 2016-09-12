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

from rampedpyrox.ratedata.ratedata_helper import(
	_calc_phi,
	_deconvolve,
	_f_phi_diff,
	_gaussian,
	_peak_indices)

from rampedpyrox.core.exceptions import(
	# rpException,
	ArrayError,
	# FileError,
	FitError,
	LengthError,
	# RunModelError,
	ScalarError,
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
	nPeaks = 4, 
	omega = 3)


class test_ratedata_helper_functions:

	def test_calc_phi(self):

		#assert peak_shape string must be "Gaussian"
		assert_raises(
			StringError,
			_calc_phi,
			[1,2,3],
			1,
			1,
			1,
			'garbage string')

		#assert that peak arrays must be same length
		assert_raises(
			LengthError,
			_calc_phi,
			[1,2,3],
			[1,2],
			[1,2],
			[1,2,3],
			'Gaussian')

		phi, peaks = _calc_phi(
			ratedata.Ea,
			[150, 200],
			[10, 10],
			1,
			'Gaussian')

		#assert shapes are right
		assert_equal(phi.shape[0], 300)
		assert_equal(peaks.shape[0], 300)
		assert_equal(peaks.shape[1], 2)

		#assert datatypes are right
		assert_equal(phi.dtype, 'float')
		assert_equal(peaks.dtype, 'float')

	def test_deconvolve_input(self):

		#assert that input types must be correct
		assert_raises(
			ScalarError,
			_deconvolve,
			ratedata.Ea,
			ratedata.f,
			nPeaks = 'garbage string',
			peak_shape = 'Gaussian',
			thres = 0.05)

		assert_raises(
			StringError,
			_deconvolve,
			ratedata.Ea,
			ratedata.f,
			nPeaks = 'auto',
			peak_shape = 'garbage string',
			thres = 0.05)

		assert_raises(
			ScalarError,
			_deconvolve,
			ratedata.Ea,
			ratedata.f,
			nPeaks = 'auto',
			peak_shape = 'Gaussian',
			thres = int(1))

		assert_raises(
			ScalarError,
			_deconvolve,
			ratedata.Ea,
			ratedata.f,
			nPeaks = 'garbage string',
			peak_shape = 'Gaussian',
			thres = 2.0)

	def test_deconvolve_output(self):

		peaks, peak_info = _deconvolve(
			ratedata.Ea,
			ratedata.f,
			nPeaks = 4,
			peak_shape = 'Gaussian',
			thres = 0.05)

		#assert shapes and dtype are right
		assert_equal(peaks.shape[0], 300)
		assert_equal(peaks.shape[1], 4)
		assert_equal(peak_info.shape[0], 4)
		assert_equal(peak_info.shape[1], 3)

		#assert peaks instance and dtype
		assert_is_instance(peaks, np.ndarray)
		assert_is_instance(peak_info, np.ndarray)
		assert_equal(peaks.dtype, 'float')

		#assert peaks are nonnegative and go to zero
		assert_almost_equal(peaks.min(axis=0).all(), 0, places=3)

		#assert peaks integrate to one
		a = np.sum(peaks, axis=1)*np.gradient(ratedata.Ea)
		assert_almost_equal(np.sum(a), 1, places=3)

	def test_deconvolve_values(self):
		#make-up a sum of gaussians
		x = np.arange(0,100)
		mu = [25, 50]
		sigma = [5, 5]
		y = _gaussian(x, mu, sigma)

		#add 5 percent relative noise
		noise = np.random.standard_normal(size = (len(x), 2))

		y = np.sum(y, axis=1) # + noise*y*0.05, axis=1)

		#deconvolve and check
		peaks, peak_info = _deconvolve(x, y)

		assert_equal(peak_info.shape[0], 2)
		assert_almost_equal(peak_info[0,0], 25, places=3)
		assert_almost_equal(peak_info[1,0], 50, places=3)

		assert_almost_equal(peak_info[0,1], 5, places=3)
		assert_almost_equal(peak_info[1,1], 5, places=3)


	def test_f_phi_diff(self):

		#assert that params must come in threes
		assert_raises(
			ArrayError,
			_f_phi_diff,
			[1,1],
			ratedata.Ea,
			ratedata.f,
			'Gaussian')

		#assert that string must be Gaussian
		assert_raises(
			StringError,
			_f_phi_diff,
			[1,1,1],
			ratedata.Ea,
			ratedata.f,
			'garbage string')

		#assert resulting array length and dtype
		d = _f_phi_diff(
			[1,1,1],
			ratedata.Ea,
			ratedata.f,
			'Gaussian')

		assert_equal(d.shape[0], 300)
		assert_equal(d.dtype, 'float')
		assert_is_instance(d, np.ndarray)

	def test_gaussian(self):

		x = np.arange(0,100,2)

		y = _gaussian(
			x,
			50,
			5)

		y2 = _gaussian(
			x,
			[25, 50, 75],
			[5, 5, 5])

		#assert output shape, type, and dtype
		assert_equal(y.shape[0], len(x))
		assert_equal(y2.shape[1], 3)
		assert_is_instance(y, np.ndarray)
		assert_equal(y.dtype, 'float')

		#assert y2 is 2d array
		assert_equal(y2.ndim, 2)

		#assert that everything integrates to one
		a = y*np.gradient(x)
		a2 = np.sum(y2, axis=1)*np.gradient(x)

		assert_almost_equal(np.sum(a), 1, places=3)
		assert_almost_equal(np.sum(a2), 3, places=3)

	def test_peak_indices(self):

		ind, lb_ind, ub_ind = _peak_indices(
			ratedata.f,
			nPeaks = 4,
			thres = 0.05)

		#assert input types
		assert_raises(
			ScalarError,
			_peak_indices,
			ratedata.f,
			nPeaks = 'garbage string',
			thres = 0.05)

		assert_raises(
			ScalarError,
			_peak_indices,
			ratedata.f,
			nPeaks = 'auto',
			thres = int(1))

		assert_raises(
			ScalarError,
			_peak_indices,
			ratedata.f,
			nPeaks = 'auto',
			thres = 2.0)

		#assert output shapes and types
		assert_is_instance(ind, np.ndarray)
		assert_equal(ind.dtype, 'int')
		assert_equal(ind.shape[0], 4)

		assert_is_instance(lb_ind, np.ndarray)
		assert_equal(lb_ind.dtype, 'int')
		assert_equal(lb_ind.shape[0], 4)

		assert_is_instance(lb_ind, np.ndarray)
		assert_equal(lb_ind.dtype, 'int')
		assert_equal(lb_ind.shape[0], 4)

		#assert that all peaks greater than thres
		vals = ratedata.f[ind]
		thres = 0.05*(np.max(ratedata.f)-np.min(ratedata.f))+np.min(ratedata.f)

		assert_equal((vals>=thres).all(), True)

		#assert fit error if nPeaks too large
		assert_raises(
			FitError,
			_peak_indices,
			ratedata.f,
			nPeaks = 20,
			thres = 0.05)

		#assert that an error is raised when there are no peaks
		assert_raises(
			FitError,
			_peak_indices,
			[1,2,3],
			nPeaks = 'auto',
			thres = 0.05)


class test_ratedata_creation:

	def test_bare_bones_input_types(self):

		#assert that manually importing data takes proper types
		#inputting string for Ea
		assert_raises(
			ArrayError, 
			rp.EnergyComplex,
			'[1,2,3]')

		#inputting negative values for Ea
		assert_raises(
			ArrayError, 
			rp.EnergyComplex,
			[-1, 0, 1, 2])

		#inputting string for f
		assert_raises(
			ArrayError, 
			rp.EnergyComplex,
			[1,2,3],
			f='[1,2,3]')

	def test_bare_bones_creation_types(self):

		ec = rp.EnergyComplex([1,2,3], f=[1,2,3])

		#asserting types are floats
		assert_equal(ec.Ea.dtype, 'float')
		assert_equal(ec.f.dtype, 'float')
		assert_equal(ec.f_std.dtype, 'float')

		#assert everything is an ndarray
		assert_is_instance(ec.Ea, np.ndarray)
		assert_is_instance(ec.f, np.ndarray)
		assert_is_instance(ec.f_std, np.ndarray)

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
			ArrayError,
			rp.EnergyComplex.inverse_model,
			model,
			timedata,
			combined = (5,6),
			omega = 3)

		assert_raises(
			ScalarError,
			rp.EnergyComplex.inverse_model,
			model,
			timedata,
			combined = [(5.0,6.0)],
			omega = 3)

		assert_raises(
			ScalarError,
			rp.EnergyComplex.inverse_model,
			model,
			timedata,
			combined = None,
			omega = [1,2,3])

	def test_input_estimated_instances(self):

		#create an inverse model to input estimated data
		ec = rp.EnergyComplex.inverse_model(
			model, 
			timedata,
			combined=[(1,2)],
			nPeaks = 5, 
			omega = 3)

		#assert all equalities
		assert_equal(ec.nPeak, 5) #STORES BEFORE COMBINING
		assert_equal(ec.dof, 286)
		assert_equal(ec.peaks.ndim, 2)
		assert_equal(ec._pkinf.ndim, 2)

		#make sure everything is the right type
		assert_is_instance(ec.peaks, np.ndarray)
		assert_is_instance(ec.resid_rmse, float)
		assert_is_instance(ec.rgh_rmse, float)
		assert_is_instance(ec._pkinf, np.ndarray)
		assert_is_instance(ec.rmse, float)

	def test_input_estimated_values(self):

		#make an EC with a single gaussian as "true" f
		x = np.arange(0,100)
		ec = rp.EnergyComplex(
			x,
			_gaussian(x, 50, 10))

		#assert that non-unity integral peaks give warning
		assert_warns(
			UserWarning,
			ec.input_estimated,
			2*_gaussian(x, 50, 10),
			omega=3)

		#assert that rmse is zero for fake data
		ec.input_estimated(_gaussian(x, 50, 10))

		assert_almost_equal(ec.rmse, 0, places=6)

if __name__ == '__main__':

	import nose

	nose.runmodule(
		argv = [__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
		exit=False)

