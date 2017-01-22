# TODO: Update examples
# TODO: Update summary
# TODO: Add plotting method


'''
This module contains the Results superclass and all corresponding subclasses.
'''

from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['RpoIsotopes']

import numpy as np
import warnings

from collections import Sequence

#import exceptions
from ..core.exceptions import(
	ArrayError,
	LengthError,
	)

#import helper functions
from ..core.core_functions import(
	assert_len,
	)

from ..core.summary_helper import(
	_calc_ri_info
	)

from .results_helper import(
	_calc_E_frac,
	_rpo_blk_corr,
	_rpo_extract_iso,
	_rpo_mass_bal_corr,
	_rpo_kie_corr,
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


class RpoIsotopes(Results):
	__doc__='''
	Class for inputting Ramped PyrOx isotopes, calculating p0(E) contained in
	each RPO fraction, correcting d13C values for kinetic fractionation, and
	storing resulting data and statistics.

	Parameters
	----------
	d13C_frac : None or array-like
		Array of the d13C values (VPDB) of each measured fraction, 
		length `nFrac`. Defaults to `None`.

	d13C_frac_std : None or array-like
		The standard deviation of `d13C_frac` with length `nFrac`. Defaults to
		zeros or `None` if `d13C_frac` is `None`.

	Fm_frac : None or array-like
		Array of the  Fm values of each measured fraction, length `nFrac`.
		Defaults to `None`.

	Fm_frac_std : None or array-like
		The standard deviation of `Fm_frac` with length `nFrac`. Defaults to
		zeros or `None` if `Fm_frac` is `None`.

	m_frac : None or array-like
		Array of the masses (ugC) of each measured fraction, length `nFrac`.
		Defaults to `None`.

	m_frac_std : None or array-like
		The standard deviation of `d13C_frac` with length `nFrac`. Defaults to
		zero or `None` if `m_frac` is `None`.

	t_frac : None or array-like
		2d array of the initial and final times of each fraction, in seconds.
		Shape [`nFrac` x 2]. Defaults to `None`.

	Warnings
	--------
	UserWarning
		If using an an isothermal model type for an RPO run.

	UserWarning
		If using a non-energy complex ratedata type for an RPO run.

	Raises
	------
	ArrayError
		If `t_frac` is not array-like.

	ArrayError
		If `nE` is not the same in the ``rp.Model`` instance and the 
		``rp.RateData`` instance.

	Notes
	-----
	When inputting `t_frac`, a time of 0 (i.e. `t0`, the initial time) is
	defined as the first timepoint in the ``RpoThermogram`` instance. If time
	passed between the thermogram `t0` and the beginning of fraction 1 
	trapping (as is almost always the case), `t_frac` must be adjusted 
	accordingly. This is done automatically when importing from .csv (see
	``RpoIsotopes.from_csv``) documenatation for info.

	See Also
	--------
	Daem
		``Model`` subclass used to generate the transform for RPO
		data and translate between time- and E-space.

	EnergyComplex
		``RateData`` subclass for storing and analyzing RPO rate data.

	RpoThermogram
		``TimeData`` subclass containing the time and fraction remaining data
		used for the inversion.

	Examples
	--------
	Generating a bare-bones isotope result instance containing only arbitrary
	time and Fm data::

		#import modules
		import rampedpyrox as rp
		import numpy as np

		#generate arbitrary data for 3 fractions
		t_frac = [[100, 200], [200, 300], [300, 1000]]
		t_frac = np.array(t_frac)

		Fm_frac = [1.0, 0.5, 0.0]

		#create instance
		ri = rp.RpoIsotopes(
			t_frac = t_frac,
			Fm_frac = Fm_frac)

	Generating a isotope result instance using an RPO output .csv file and the
	``RpoIsotopes.from_csv`` class method::

		#import modules
		import rampedpyrox as rp

		#create path to data file
		file = 'path_to_folder_containing_data/isotope_data.csv'

		#create instance
		ri = rp.RpoThermogram.from_csv(
			file,
			blk_corr = True,
			mass_err = 0.01)

	This automatically corrected inputted isotopes for the inputted instrument
	blank carbon contribution using the `blk_corr` flag and assumed a 1\% 
	uncertainty in mass measurements. **NOTE:** See ``RpoIsotopes.from_csv`` 
	documentation for instructions on getting the .csv file in the right
	format.

	Correcting the d13C results for a given E distribution for kinetic isotope
	fractionation effects::

		#assuming there exists some Daem, EnergyComplex, and 
		# RpoThermogram instances already created

		ri.d13C_correct(
			daem, 
			ec, 
			tg, 
			DEa = 0.0018)

	Printing a summary of the analysis::

		#fraction and component information
		print(ri.frac_info)

		#RMSE values
		m = 'mass RMSE (ugC): %.2f' %ri.m_rmse
		d13C = 'd13C RMSE (VPDB): %.2f' %ri.d13C_rmse
		Fm = 'Fm RMSE: %.4f' %ri.Fm_rmse

		print(m+'\\n'+d13C+'\\n'+Fm)

	**Attributes**

	d13C_frac : np.ndarray
		Array of the d13C values (VPDB) of each measured fraction, 
		length `nFrac`.

	d13C_frac_std : np.ndarray
		The standard deviation of `d13C_frac` with length `nFrac`.

	d13C_corr : np.ndarray
		Array of the d13C values (VPDB) of each measured fraction, corrected
		for kinetic isotope fractionation. Length `nFrac`.

	d13C_corr_std : np.ndarray
		The standard deviation of the d13C values (VPDB) of each measured 
		fraction, corrected for kinetic isotope fractionation. Length `nFrac`.

	E_frac : np.ndarray
		Array of the mean E value (kJ) contained in each measured fraction as
		calculated by the inverse model, length `nFrac`.

	E_frac_std : np.ndarray
		The standard deviation of E (kJ) contained in each measured fraction
		as calculated by the inverse model, length `nFrac`.

	Fm_frac : np.ndarray
		Array of the Fm values of each measured fraction, length `nFrac`.

	Fm_frac_std : np.ndarray
		The standard deviation of `Fm_frac` with length `nFrac`.

	frac_info : pd.DataFrame
		Dataframe containing the inputted fraction isotope summary info: 

			time (init. and final), \n
			mass (mean and std.), \n
			d13C (mean and std.), \n
			d13C corrected (mean and std.), \n
			Fm (mean and std.), \n
			E (mean and std.) \n

	m_frac : np.ndarray
		Array of the masses (ugC) of each measured fraction, length `nFrac`.

	m_frac_std : np.ndarray
		The standard deviation of `m_frac` with length `nFrac`.

	nFrac : int
		The number of measured fractions.

	t_frac : np.ndarray
		2d array of the initial and final times of each fraction, in seconds.
		Shape [`nFrac` x 2].
	'''

	def __init__(
			self,
			model,
			ratedata,
			t_frac,
			d13C_raw = None, 
			d13C_raw_std = None, 
			Fm_raw = None,
			Fm_raw_std = None,
			m_raw = None,
			m_raw_std = None,
			blk_corr = False,
			mb_corr = False,
			kie_corr = False):

		#assert all lenghts are the same and of the same length as t_frac, and
		# store as attributes
		if isinstance(t_frac, str):
			raise ArrayError(
				't_frac cannot be a string')

		elif isinstance(t_frac, Sequence) or hasattr(t_frac, '__array__'):
			
			n = len(t_frac)
			self.t_frac = t_frac
			self.nFrac = n

		else:
			raise ArrayError(
				't_frac must be array-like')

		#store existing data
		if m_raw is not None:
			self.m_raw = assert_len(m_raw, n)

			#store stdev if it exists, zeros if not
			if m_raw_std is not None:
				self.m_raw_std = assert_len(m_raw_std, n)
			else:
				self.m_raw_std = assert_len(0, n)

		if d13C_raw is not None:
			self.d13C_raw = assert_len(d13C_raw, n)

			#store stdev if it exists, zeros if not
			if d13C_raw_std is not None:
				self.d13C_raw_std = assert_len(d13C_raw_std, n)
			else:
				self.d13C_raw_std = assert_len(0, n)

		if Fm_raw is not None:
			self.Fm_raw = assert_len(Fm_raw, n)

			#store stdev if it exists, zeros if not
			if Fm_raw_std is not None:
				self.Fm_raw_std = assert_len(Fm_raw_std, n)
			else:
				self.Fm_raw_std = assert_len(0, n)

		#warn if model is not Daem
		mod_type = type(model).__name__

		if mod_type not in ['Daem']:
			warnings.warn(
				'Attempting to calculate E distributions using a model' 
				' instance of type %r. Consider using rp.Daem instance'
				' instead' % mod_type, UserWarning)

		#warn if ratedata is not EnergyComplex
		rd_type = type(ratedata).__name__

		if rd_type not in ['EnergyComplex']:
			warnings.warn(
				'Attempting to calculate E distributions using a ratedata'
				' instance of type %r. Consider using rp.EnergyComplex'
				' instance instead' % rd_type, UserWarning)

		#raise exception if not the right shape
		if model.nE != ratedata.nE:
			raise ArrayError(
				'Cannot combine model with nE = %r and RateData with'
				' nE = %r. Check that RateData was not created using a'
				' different model' % (model.nE, ratedata.nE))

		#calculate E distributions for each fraction
		E_frac, E_frac_std, p_frac = _calc_E_frac(self, model, ratedata)
		
		#store results
		self.E_frac = E_frac
		self.E_frac_std = E_frac_std
		self._p_frac = p_frac

		#store raw info summary table
		self.ri_raw_info = _calc_ri_info(self, flag = 'raw')

		#store bookkeeping corrected flags
		self._blk_corr = blk_corr
		self._mb_corr = mb_corr
		self._kie_corr = kie_corr

	#define classmethod for creating instance and populating measured values
	# directly from a .csv file
	@classmethod
	def from_csv(
			cls, 
			file,
			model,
			ratedata,
			blk_corr = False,
			blk_d13C = (-29.0, 0.1),
			blk_flux = (0.375, 0.0583),
			blk_Fm =  (0.555, 0.042),
			bulk_d13C_true = None,
			DE = 0.0018,
			mass_err = 0.01):
		'''
		Class method to directly import RPO fraction data from a .csv file and
		create an ``RpoIsotopes`` class instance.
		
		Parameters
		----------
		model : rp.Model
			``rp.Model`` instance containing the A matrix to use for 
			inversion.

		ratedata : rp.RateData
			``rp.Ratedata`` instance containing the reactive continuum data.
			file : str or pd.DataFrame
				File containing isotope data, either as a path string or a
				dataframe.

		blk_corr : Boolean
			Tells the method whether or not to blank-correct isotope data. If
			`True`, blank-corrects according to inputted blank composition 
			values. If `bulk_d13C_true` is not `None`, further corrects d13C 
			values to ensure isotope mass balance (see Hemingway et al.,
			Radiocarbon **2017** for details).

		blk_d13C : tuple
			Tuple of the blank d13C composition (VPDB), in the form 
			(mean, stdev.) to be used of ``blk_corr = True``. Defaults to the
			NOSAMS RPO blank as calculated by Hemingway et al., Radiocarbon
			**2017**.

		blk_flux : tuple
			Tuple of the blank flux (ng/s), in the form (mean, stdev.) to
			be used of ``blk_corr = True``. Defaults to the NOSAMS RPO blank 
			as calculated by Hemingway et al., Radiocarbon **2017**.

		blk_Fm : tuple
			Tuple of the blank Fm value, in the form (mean, stdev.) to
			be used of ``blk_corr = True``. Defaults to the NOSAMS RPO blank 
			as calculated by Hemingway et al., Radiocarbon **2017**.

		bulk_d13C_true : None or array
			True measured d13C value (VPDB) for bulk material as measured
			independently (e.g. on a EA-IRMS). If not `None`, this value is
			used to mass-balance-correct d13C values as described in Hemingway
			et al., Radiocarbon **2017**. If not `none`, must be inputted in
			the form [mean, stdev.]

		DE : scalar
			Value for the difference in E between 12C- and 13C-containing
			atoms, in kJ. Defaults to 0.0018 (the best-fit value calculated
			in Hemingway et al., **2017**).

		mass_err : float
			Relative uncertainty in mass measurements, typically as a sum of
			manometric uncertainty in pressure measurements and uncertainty in
			vacuum line volumes. Defaults to 0.01 (i.e. 1\% relative 
			uncertainty).
		
		Notes
		-----
		For bookkeeping purposes, the first 2 rows must be fractions "-1" and
		"0", where the timestamp for fraction "-1" is the first point in 
		the `all_data` file used to create the ``rp.RpoThermogram`` instance,
		and the timestamp for fraction "0" is the t0 for the first fraction.

		If mass, d13C, and Fm data exist, column names must be the following:

			'ug_frac' and 'ug_frac_std' \n
			'd13C' and 'd13C_std' \n
			'Fm' and 'Fm_std'

		See Also
		--------
		RpoThermogram.from_csv
			Classmethod for creating ``rp.RpoThermogram`` instance directly 
			from a .csv file.

		References
		----------
		[1] J.D. Hemingway et al. (2017) Assessing the blank carbon
			contribution, isotope mass balance, and kinetic isotope 
			fractionation of the ramped pyrolysis/oxidation instrument at 
			NOSAMS. *Radiocarbon*
		'''

		#extract data from file
		d13C, d13C_std, Fm, Fm_std, m, m_std, t_frac = _rpo_extract_iso(
			file,
			mass_err)

		#create RpoIsotopes instance and store raw data
		ri = cls(
			model,
			ratedata,
			t_frac,
			d13C_raw = d13C,
			d13C_raw_std = d13C_std,
			Fm_raw = Fm,
			Fm_raw_std = Fm_std,
			m_raw = m,
			m_raw_std = m_std,
			blk_corr = False,
			mb_corr = False,
			kie_corr = False)

		#blank correct m, d13C, Fm if necessary
		if blk_corr is True:

			#call blank- and mass-balance correction method
			ri.blank_correct(
				blk_d13C = blk_d13C,
				blk_flux = blk_flux,
				blk_Fm =  blk_Fm,
				bulk_d13C_true = bulk_d13C_true)

		#kinetic fractionation correct if necessary
		if DE is not None:

			#call kie correction method
			ri.kie_correct(
				model,
				ratedata,
				DE = DE)

		return ri

	def blank_correct(
		self,
		blk_d13C = (-29.0, 0.1),
		blk_flux = (0.375, 0.0583),
		blk_Fm =  (0.555, 0.042),
		bulk_d13C_true = None):
		'''
		Method to blank- and mass-balance correct raw isotope values.

		Parameters
		----------

		Raises
		--------
		UserWarning
			If already corrected for blank contribution
		
		UserWarning
			If already corrected for 13C mass balance

		References
		----------
		[1] J.D. Hemingway et al. (2017) Assessing the blank carbon
			contribution, isotope mass balance, and kinetic isotope 
			fractionation of the ramped pyrolysis/oxidation instrument at 
			NOSAMS. *Radiocarbon*
		'''

		#raise warnings
		if self._blk_corr == True:
			warnings.warn(
				'd13C, Fm, and/or m has already been blank-corrected!'
				' Proceeding anyway', UserWarning)

		if self._mb_corr == True:
			warnings.warn(
				'd13C has already been mass-balance corrected!'
				' Proceeding anyway', UserWarning)

		#extract d13C from self to be corrected
		if hasattr(self, d13C_corr):
			d13C = self.d13C_corr
			d13C_std = self.d13C_corr_std
		
		else:
			d13C = self.d13C_raw
			d13C_std = self.d13C_raw_std

		#extract Fm from self to be corrected
		if hasattr(self, Fm_corr):
			Fm = self.Fm_corr
			Fm_std = self.Fm_corr_std
		
		else:
			Fm = self.Fm_raw
			Fm_std = self.Fm_raw_std

		#extract m from self to be corrected
		if hasattr(self, m_corr):
			m = self.m_corr
			m_std = self.m_corr_std
		
		else:
			m = self.m_raw
			m_std = self.m_raw_std

		#blank-correct values
		d13C, d13C_std, Fm, Fm_std, m, m_std = _rpo_blk_corr(
			d13C,
			d13C_std,
			Fm,
			Fm_std,
			m,
			m_std,
			t,
			blk_d13C = blk_d13C,
			blk_flux = blk_flux,
			blk_Fm = blk_Fm)

		#set bookkeeping flag
		self._blk_corr = True

		#mass-balance correct d13C if necessary
		if bulk_d13C_true is not None:

			d13C, d13C_std = _rpo_mass_bal_corr(
				d13C,
				d13C_std,
				m,
				m_std,
				bulk_d13C_true)

			#set bookkeeping flag
			self._mb_corr = True

		#store corrected values if they exist
		if m is not None:
			self.m_corr = assert_len(m, n)

			#store stdev if it exists, zeros if not
			if m_std is not None:
				self.m_corr_std = assert_len(m_std, n)
			else:
				self.m_corr_std = assert_len(0, n)

		if d13C is not None:
			self.d13C_corr = assert_len(d13C, n)

			#store stdev if it exists, zeros if not
			if d13C_corr_std is not None:
				self.d13C_corr_std = assert_len(d13C_std, n)
			else:
				self.d13C_corr_std = assert_len(0, n)

		if Fm is not None:
			self.Fm_corr = assert_len(Fm_corr, n)

			#store stdev if it exists, zeros if not
			if Fm_std is not None:
				self.Fm_corr_std = assert_len(Fm_std, n)
			else:
				self.Fm_corr_std = assert_len(0, n)

		#generate summary table
		self.ri_corr_info = _calc_ri_info(self, flag = 'corr')

	def kie_correct(
		self,
		model,
		ratedata,
		DE = 0.0018):
		'''
		Method for further correcting d13C values to account for kinetic 
		isotope fractionation occurring within the instrument.

		Parameters
		----------
		model : rp.Model
			``rp.Model`` instance containing the A matrix to use for 
			inversion.

		ratedata : rp.RateData
			``rp.Ratedata`` instance containing the reactive continuum data.

		DE : scalar
			Value for the difference in E between 12C- and 13C-containing
			atoms, in kJ. Defaults to 0.0018 (the best-fit value calculated
			in Hemingway et al., **2017**).

		Warnings
		--------
		UserWarning
			If already corrected for kinetic fractionation

		References
		----------
		[1] J.D. Hemingway et al. (2017) Assessing the blank carbon
			contribution, isotope mass balance, and kinetic isotope 
			fractionation of the ramped pyrolysis/oxidation instrument at 
			NOSAMS. *Radiocarbon*
		'''

		#raise warnings
		if self._kie_corr == True:
			warnings.warn(
				'd13C has already been corrected for kinetic fractionation!'
				' Proceeding anyway', UserWarning)

		#extract d13C from self to be corrected
		if hasattr(self, d13C_corr):
			d13C = self.d13C_corr
			d13C_std = self.d13C_corr_std
		
		else:
			d13C = self.d13C_raw
			d13C_std = self.d13C_raw_std

		#kie-correct values
		d13C, d13C_std = _rpo_kie_corr(
			self,
			d13C,
			d13C_std,
			model,
			ratedata,
			DE = DE)

		#set bookkeeping flag
		self._kie_corr = True

		#store results
		if d13C is not None:
			self.d13C_corr = assert_len(d13C, n)

			#store stdev if it exists, zeros if not
			if d13C_corr_std is not None:
				self.d13C_corr_std = assert_len(d13C_std, n)
			else:
				self.d13C_corr_std = assert_len(0, n)

		#store summary
		self.ri_corr_info = _calc_ri_info(self, flag = 'corr')

	def plot(ax1, ax2, iso = 'Fm', plot_corr = True):






if __name__ == '__main__':

	import rampedpyrox as rp
