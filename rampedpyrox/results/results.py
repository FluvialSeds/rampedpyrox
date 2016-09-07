'''
This module contains the Results superclass and all corresponding subclasses.
'''

import numpy as np
import warnings

from collections import Sequence
from scipy.optimize import nnls

#import helper functions
from rampedpyrox.core.core_functions import(
	assert_len,
	)

from rampedpyrox.results.results_helper import(
	_kie_d13C,
	_rpo_blk_corr,
	_rpo_cont_ptf,
	_rpo_extract_iso,
	)

from rampedpyrox.core.summary_helper import(
	_rpo_isotopes_frac_info,
	# _rpo_isotopes_peak_info,
	)

class Results(object):
	'''
	Class to store resulting data (e.g. Rpo isotopes, pyGC composition, etc.).
	Intended for subclassing. Do not call directly.
	'''

	def __init__(self):
		'''
		Initialize the superclass
		'''
		raise NotImplementedError

	#define classmethod for creating instance and populating measured values
	# directly from a .csv file
	@classmethod
	def from_csv(cls, file):
		raise NotImplementedError

	#define method for fitting result data from a TimeData instance
	def fit(self, timedata):
		raise NotImplementedError


class RpoIsotopes(Results):
	__doc__='''
	Class for inputting Rpo isotopes, calculating estimated peak isotope 
	values, and storing corresponding statistics.

	Parameters
	----------

	Keyword Arguments
	-----------------

	Raises
	------

	Warnings
	--------

	Notes
	-----

	See Also
	--------

	Examples
	--------

	Attributes
	----------
	d13C_frac : np.ndarray

	d13C_frac_std : np.ndarray

	Fm_frac : np.ndarray

	Fm_frac_std : np.ndarray

	frac_info : pd.DataFrame

	m_frac : np.ndarray

	m_frac_std : np.ndarray

	nFrac : int

	t_frac : np.ndarray

	References
	----------
	'''

	def __init__(self, d13C_frac = None, d13C_frac_std = 0, Fm_frac = None,
		Fm_frac_std = 0, m_frac = None, m_frac_std = 0, t_frac = None):

		#assert all lenghts are the same and of the same length as t_frac, and
		# store as attributes
		if t_frac is not None:
			if isinstance(t_frac, str):
				raise TypeError('t_frac cannot be a string')

			elif isinstance(t_frac,Sequence) or hasattr(t_frac,'__array__'):
				
				n = len(t_frac)
				self.t_frac = t_frac
				self.nFrac = n

			else:
				raise TypeError('t_frac must be array-like or None')

			#store existing data
			if m_frac is not None:
				self.m_frac = assert_len(m_frac, n)
				self.m_frac_std = assert_len(m_frac_std, n)

			if d13C_frac is not None:
				self.d13C_frac = assert_len(d13C_frac, n)
				self.d13C_frac_std = assert_len(d13C_frac_std, n)

			if Fm_frac is not None:
				self.Fm_frac = assert_len(Fm_frac, n)
				self.Fm_frac_std = assert_len(Fm_frac_std, n)

			#store in frac_info attribute
			self.frac_info = _rpo_isotopes_frac_info(self)

		#store self._corrected for blank correction bookkeeping
		self._corrected = False

	#define classmethod for creating instance and populating measured values
	# directly from a .csv file
	@classmethod
	def from_csv(cls, file, blk_corr = False, mass_err = 0.01):
		'''
		Class method to directly import RPO fraction data from a .csv file and
		create an ``RpoIsotopes`` class instance.
		
		Parameters
		----------
		file : str or pd.DataFrame
			File containing isotope data, either as a path string or 
			``pd.DataFrame`` instance.

		Keyword Arguments
		-----------------
		blk_corr : Boolean
			Tells the method whether or not to blank-correct isotope data. If
			`True`, blank-corrects according to NOSAMS RPO blank as calculated
			by Hemingway et al. **(in prep)**.

		mass_err : float
			Relative uncertainty in mass measurements. Defaults to 0.01
			(i.e. 1\% relative uncertainty).

		Raises
		------
		TypeError
			If `file` is not str or ``pd.DataFrame``.
		
		TypeError
			If index is not ``pd.DatetimeIndex`` instance.	

		TypeError
			If `mass_err` is not scalar.

		ValueError
			If `file` does not contain "d13C", "d13C_std", "Fm", "Fm_std", 
			"ug_frac", and "fraction" columns.
		
		ValueError
			If first two rows are not fractions "-1" and "0"
		
		Notes
		-----
		For bookkeeping purposes, the first 2 rows must be fractions "-1" and
		"0", where the timestamp for fraction "-1" is the first point in 
		`all_data` and the timestamp for fraction "0" is the t0 for the first 
		fraction.

		See Also
		--------
		RpoThermogram.from_csv
			Classmethod for creating ``rp.RpoThermogram`` instance directly 
			from a .csv file.

		References
		----------
		J.D. Hemingway et al. **(in prep)** Assessing the blank carbon
		contribution, isotope mass balance, and kinetic isotope fractionation 
		of the ramped pyrolysis/oxidation instrument at NOSAMS.
		'''

		#estract data from file
		d13C, d13C_std, Fm, Fm_std, m, m_std, t = _rpo_extract_iso(file,
			mass_err)

		#blank correct if necessary
		if blk_corr:
			d13C, d13C_std, Fm, Fm_std, m, m_std = _rpo_blk_corr(d13C, 
				d13C_std, Fm, Fm_std, m, m_std, t)

		ri = cls(d13C_frac = d13C, d13C_frac_std = d13C_std, Fm_frac = Fm,
			Fm_frac_std = Fm_std, m_frac = m, m_frac_std = m_std, t_frac = t)

		return ri

	def fit(self, model, ratedata, timedata, DEa = None):
		'''
		Method for fitting results instance to calculate the isotope
		composition of each peak in a timedata instance.

		Parameters
		----------

		Keyword Arguments
		-----------------

		Raises
		------

		Warnings
		--------
		Warns if nPeak is greater than nFrac, the problem is underconstrained.


		Notes
		-----

		See Also
		--------

		References
		----------

		'''

		#warn if timedata is not RpoThermogram

		#raise exception if timedata does not have fitted data attributes

		#raise exception if DEa is not int or array-like with length nPeak
		if DEa is None:
			DEa = assert_len(0, ratedata.nPeak)
		
		else:
			try:
				DEa = assert_len(DEa, ratedata.nPeak)
			
			except ValueError:
				DEa = assert_len(DEa, timedata.nPeak)

				#add DEa for deleted peaks so that len(DEa) = ratedata.nPeak
				dp = [val - i for i, val in enumerate(ratedata._cmbd)]
				dp = np.array(dp)
				DEa = np.insert(DEa, dp, DEa[dp-1])

		#calculate peak contribution to each fraction
		cont_ptf, ind_min, ind_max, ind_wgh = _rpo_cont_ptf(self, timedata)

		#if has mass, calculate peak mass
		if hasattr(self, 'm_frac'):

			res_mass = nnls(cont_ptf, self.m_frac)

			self.m_peak = res_mass[0]
			self.m_rmse = res_mass[1]/(self.nFrac**0.5)

		#if has Fm, calculate peak Fm
		if hasattr(self, 'Fm_frac'):

			res_14 = nnls(cont_ptf, self.Fm_frac)
			
			self.Fm_peak = res_14[0]
			self.Fm_rmse = res_14[1]/(self.nFrac**0.5)

		#if has d13C, calculate peak d13C
		if hasattr(self, 'd13C_frac'):

			res_13 = _kie_d13C(DEa, ind_wgh, model, self, ratedata)

			self.d13C_peak = res_13[0]
			self.d13C_rmse = res_13[1]/(self.nFrac**0.5)

		#store results in summary table














