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
	RunModelError,
	)

#import helper functions
from ..core.core_functions import(
	assert_len,
	)

from ..core.summary_helper import(
	_rpo_isotopes_frac_info,
	_rpo_isotopes_peak_info,
	)

from .results_helper import(
	_d13C_to_R13,
	_kie_d13C,
	_kie_d13C_MC,
	_nnls_MC,
	_R13_CO2,
	_R13_to_d13C,
	_rpo_blk_corr,
	_rpo_cont_ptf,
	_rpo_extract_iso,
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

	Raises
	------
	ArrayError
		If `t_frac` is not array-like or `None`.

	Notes
	-----
	When inputting `t_frac`, a time of 0 (i.e. `t0`, the initial time) is
	defined as the first timepoint in the ``RpoThermogram`` instance used to
	generate the peaks of interest. If time passed between the thermogram `t0`
	and the beginning of fraction 1 trapping (as is almost always the case),
	`t_frac` must be adjusted accordingly.

	See Also
	--------
	Daem
		``Model`` subclass used to generate the Laplace transform for RPO
		data and translate between time- and Ea-space.

	EnergyComplex
		``RateData`` subclass for storing, deconvolving, and analyzing RPO
		rate data.

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
		t_frac = [[1000, 200], [200, 300], [300, 1000]]
		t_frac = np.array(t_frac)

		Fm_frac = [1.0, 0.5, 0.0]

		#create instance
		ri = rp.RpoIsotopes(t_frac = t_frac,
							Fm_frac = Fm_frac)

	Generating a isotope result instance using an RPO output .csv file and the
	``RpoIsotopes.from_csv`` class method::

		#import modules
		import rampedpyrox as rp

		#create path to data file
		file = 'path_to_folder_containing_data/isotope_data.csv'

		#create instance
		ri = rp.RpoThermogram.from_csv(file,
										blk_corr = True,
										mass_err = 0.01)

	This automatically corrected inputted isotopes for the NOSAMS instrument
	blank carbon contribution using the `blk_corr` flag and assumed a 1\% 
	uncertainty in mass measurements. **NOTE:** See ``from_csv`` documentation
	for instructions on getting the .csv file in the right format.

	Fitting the isotope results for a given Ea distribution and bootstrapping
	the uncertainty::

		#assuming there exists some Daem, EnergyComplex, and 
		# RpoThermogram instances already created

		ri.fit(daem, ec, tg, 
				DEa = None,
				nIter = 10000)

	Same as above, but now setting a constant value for `DEa` to include
	kinetic isotope fractionation::

		ri.fit(daem, ec, tg, 
				DEa = 0.0018, #value estimated for NOSAMS
				nIter = 10000)

	Additionally, each peak can be given a different `DEa` value::

		#assuming there are 5 peaks (after combining) and arbitrarily
		# making DEa values
		DEa = [0., 0.001, 0., 0.005, 0.02]

		ri.fit(daem, ec, tg, 
				DEa = DEa,
				nIter = 10000)

	If d13C data exist, the d13C value of instantaneously produced CO2 can be
	plotted against time::

		#import additional modules
		import matplotlib.pyplot as plt

		#create figure
		fig, ax = plt.subplots(1,1)

		#plot data
		ax.plot(tg.t, ri.d13C_product,
				linewidth = 2,
				color = 'k')

		#label axes
		ax.set_xlabel('time (s)')
		ax.set_ylabel(r'instantaneous $\\delta^{13}C$')

	Printing a summary of the analysis::

		#fraction and peak information
		print(ri.frac_info)
		print(ri.peak_info)

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

	d13C_peak : np.ndarray
		Array of the d13C values (VPDB) of each modeled peak (treating
		combined peaks as one), length `nPeak` (post-combining).

	d13C_peak_std : np.ndarray
		The standard deviation of `d13C_peak` with length `nPeak` 
		(post-combining).

	d13C_product : np.ndarray
		The d13C values (VPDB) of instantaneously produced product at each
		timepoint in ``ratedata.t``. Length `nt`.

	d13C_rmse : float
		The RMSE between the true and estimated d13C values of each fraction,
		in VPDB.

	Fm_frac : np.ndarray
		Array of the Fm values of each measured fraction, length `nFrac`.

	Fm_frac_std : np.ndarray
		The standard deviation of `Fm_frac` with length `nFrac`.

	Fm_peak : np.ndarray
		Array of the Fm values of each modeled peak (treating combined peaks 
		as one), length `nPeak` (post-combining).

	Fm_peak_std : np.ndarray
		The standard deviation of `Fm_peak` with length `nPeak` (post-
		combining).

	Fm_rmse : float
		The RMSE between the true and estimated Fm values of each fraction.

	frac_info : pd.DataFrame
		Dataframe containing the inputted fraction isotope summary info: 

			time (init. and final), \n
			mass (mean and std.), \n
			d13C (mean and std.), \n
			Fm (mean and std.), \n

	m_frac : np.ndarray
		Array of the masses (ugC) of each measured fraction, length `nFrac`.

	m_frac_std : np.ndarray
		The standard deviation of `m_frac` with length `nFrac`.

	m_peak : np.ndarray
		Array of the masses (ugC) of each modeled peak (treating combined 
		peaks as one), length `nPeak` (post-combining).

	m_peak_std : np.ndarray
		The standard deviation of `m_peak` with length `nPeak` (post-
		combining).

	m_rmse : float
		The RMSE between the true and estimated masses of each fraction.

	nFrac : int
		The number of measured fractions.

	nIter : int
		The number of iterations, used for bootstrapping peak mass/isotope
		uncertainty.

	peak_info : pd.DataFrame
		Dataframe containing the inverse-modeled peak isotope summary info: 

			mass (mean and std.), \n
			d13C (mean and std.), \n
			Fm (mean and std.), \n
			DEa
		
		Combined peak info is repeated with an asterisk (*) next to the 
		repeated row indices.

	t_frac : np.ndarray
		2d array of the initial and final times of each fraction, in seconds.
		Shape [`nFrac` x 2].
	'''

	def __init__(
			self, 
			d13C_frac = None, 
			d13C_frac_std = 0, 
			Fm_frac = None,
			Fm_frac_std = 0, 
			m_frac = None, 
			m_frac_std = 0, 
			t_frac = None):

		#assert all lenghts are the same and of the same length as t_frac, and
		# store as attributes
		if t_frac is not None:
			if isinstance(t_frac, str):
				raise ArrayError(
					't_frac cannot be a string')

			elif isinstance(t_frac, Sequence) or hasattr(t_frac, '__array__'):
				
				n = len(t_frac)
				self.t_frac = t_frac
				self.nFrac = n

			else:
				raise ArrayError(
					't_frac must be array-like or None')

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
			File containing isotope data, either as a path string or a
			dataframe.

		blk_corr : Boolean
			Tells the method whether or not to blank-correct isotope data. If
			`True`, blank-corrects according to NOSAMS RPO blank as calculated
			by Hemingway et al. **(in prep)**.

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
		[1] J.D. Hemingway et al. **(in prep)** Assessing the blank carbon
			contribution, isotope mass balance, and kinetic isotope 
			fractionation of the ramped pyrolysis/oxidation instrument at 
			NOSAMS.
		'''

		#estract data from file
		d13C, d13C_std, Fm, Fm_std, m, m_std, t = _rpo_extract_iso(
			file,
			mass_err)

		#blank correct if necessary
		if blk_corr:
			d13C, d13C_std, Fm, Fm_std, m, m_std = _rpo_blk_corr(
				d13C, 
				d13C_std, 
				Fm, 
				Fm_std, 
				m, 
				m_std, 
				t)

		ri = cls(
			d13C_frac = d13C, 
			d13C_frac_std = d13C_std, 
			Fm_frac = Fm,
			Fm_frac_std = Fm_std, 
			m_frac = m, 
			m_frac_std = m_std, 
			t_frac = t)

		#store 'corrected' if necessary (for bookkeeping)
		if blk_corr:
			ri._corrected = True

		return ri

	def fit(self, model, ratedata, timedata, DEa = None, nIter = 1):
		'''
		Method for fitting ``RpoResults`` instance in order to calculate the 
		isotope composition of each inverse-modeled peak in an 
		``RpoThermogram`` instance.

		Parameters
		----------
		model : rp.Model
			``rp.Model`` instance containing the A matrix to use for 
			inversion.

		ratedata : rp.RateData
			``rp.Ratedata`` instance containing the reactive continuum data.

		timedata : rp.TimeData
			``rp.TimeData`` instance containing the estimated timeseries data.

		DEa : None, scalar, or array-like
			The difference in Ea between 12C- and 13C-containing molecules
			within each Ea peak, in kJ/mol. If `None`, no kinetic isotope 
			effect is included. If scalar, a single value will be used for 
			all Ea peaks.

		nIter : int
			The number of iterations, used for bootstrapping peak mass/isotope
			uncertainty.

		Raises
		------
		RunModelError
			If ratedata does not contain attribute 'peaks' -- i.e. if the
			inverse model has not been run.

		RunModelError
			If timedata does not contain attribute 'cmpt' -- i.e. if the
			forward model has not been run.

		Warnings
		--------
		UserWarning
			If nPeak is greater than nFrac, the problem is underconstrained.

		UserWarning
			If attempting to use timedata that is not a ``rp.RpoThermogram``
			instance.

		UserWarning
			If attempting to use ratedata that is not a ``rp.EnergyComplex``
			instance.

		UserWarning
			If attempting to use a model that is not a ``rp.Daem`` instance.

		UserWarning
			If ``scipy.optimize.least_squares`` cannot converge on a
			successful fit when estimating d13C values for each peak.

		Notes
		-----
		When analyzing Ramped PyrOx thermograms, peak masses (and mass RMSE)
		are calculated by comparing the peak shapes (a function of the 
		thermogram) to inputted fraction masses, which are typically measured
		manometrically. Thus, mass RMSE serves as a metric for the agreement 
		between photometically and manometrically determined masses (see 
		Rosenheim et al., 2008).

		Attempting to deconvolve a thermogram containing multiple peaks that 
		(nearly) exclusively reside within a single fraction will lead to
		spurrious results such as 0 masses and wildly varying d13C values.
		Consider combining peaks until the problem is overconstrained.

		References
		----------
		[1] B. Cramer (2004) Methane generation from coal during open system 
			pyrolysis investigated by isotope specific, Gaussian distributed 
			reaction kinetics. *Organic Geochemistry*, **35**, 379-392.

		[2] Rosenheim et al. (2008) Antarctic sediment chronology by 
			programmed-temperature pyrolysis: Methodology and data treatment. 
			*Geochemistry, Geophysics, Geosystems*, **9(4)**, GC001816.
		'''

		#warn if model is not Daem
		mod_type = type(model).__name__

		if mod_type not in ['Daem']:
			warnings.warn(
				'Attempting to calculate isotopes using a model instance of'
				' type %r. Consider using rp.Daem instance instead'
				% rd_type)

		#warn if ratedata is not EnergyComplex
		rd_type = type(ratedata).__name__

		if rd_type not in ['EnergyComplex']:
			warnings.warn(
				'Attempting to calculate isotopes using a ratedata instance of'
				' type %r. Consider using rp.EnergyComplex instance instead'
				% rd_type)

		#warn if timedata is not RpoThermogram
		td_type = type(timedata).__name__

		if td_type not in ['RpoThermogram']:
			warnings.warn(
				'Attempting to calculate isotopes using an isothermal timedata'
				' instance of type %r. Consider using rp.RpoThermogram' 
				' instance instead' % td_type)

		#raise exception if timedata does not have fitted data attributes
		if not hasattr(timedata, 'cmpt'):
			raise RunModelError(
				'timedata instance must have attribute "cmpt". Run the'
				' forward model before solving for isotopes!')

		#raise exception if ratedata does not have fitted data attributes
		if not hasattr(ratedata, 'peaks'):
			raise RunModelError(
				'ratedata instance must have attribute "peaks". Run the'
				' inverse model before solving for isotopes!')

		#raise exception if DEa is not int or array-like with length nPeak
		if DEa is None:
			DEa = assert_len(0, ratedata.nPeak)
		
		else:
			try:
				DEa = assert_len(DEa, ratedata.nPeak)
			
			except LengthError:

					#add DEa for deleted peaks so len(DEa) = ratedata.nPeak
					dp = [val - i for i, val in enumerate(ratedata._cmbd)]
					dp = np.array(dp)
					DEa = np.insert(DEa, dp, DEa[dp-1])

					#assert length is nPeak (before combining)
					DEa = assert_len(DEa, timedata.nPeak)

		#calculate peak contribution to each fraction
		cont_ptf, ind_min, ind_max, ind_wgh = _rpo_cont_ptf(
			self, 
			timedata, 
			ptf = True)

		cont_ftp, _, _, _ = _rpo_cont_ptf(
			self, 
			timedata, 
			ptf = False) #frac to peak for mass calc.

		#solve each isotope/mass, and calculate Monte Carlo uncertainty
		nIter = int(nIter)
		self.nIter = nIter

		#if has mass, calculate peak mass
		if hasattr(self, 'm_frac'):

			m_peak, m_peak_std, m_rmse = _nnls_MC(
				cont_ftp, 
				nIter, 
				self.m_frac, 
				self.m_frac_std)

			self.m_peak = m_peak
			self.m_peak_std = m_peak_std
			self.m_rmse = m_rmse

		#if has Fm, calculate peak Fm
		if hasattr(self, 'Fm_frac'):

			Fm_peak, Fm_peak_std, Fm_rmse = _nnls_MC(
				cont_ptf, 
				nIter, 
				self.Fm_frac, 
				self.Fm_frac_std)

			self.Fm_peak = Fm_peak
			self.Fm_peak_std = Fm_peak_std
			self.Fm_rmse = Fm_rmse

		#if has d13C, calculate peak d13C and instantaneous CO2 d13C
		if hasattr(self, 'd13C_frac'):

			d13C_peak, d13C_peak_std, d13C_rmse = _kie_d13C_MC(
				DEa, 
				ind_wgh, 
				model, 
				nIter, 
				self, 
				ratedata)

			self.d13C_peak = d13C_peak
			self.d13C_peak_std = d13C_peak_std
			self.d13C_rmse = d13C_rmse

			#calculate d13C of CO2 at each timepoint
			R13_peak = _d13C_to_R13(d13C_peak)
			R13_CO2 = _R13_CO2(DEa, model, R13_peak, ratedata)
			d13C_CO2 = _R13_to_d13C(R13_CO2)

			self.d13C_product = d13C_CO2

		#store results in summary table
		self.peak_info = _rpo_isotopes_peak_info(
			ratedata._cmbd, 
			DEa, 
			self)

if __name__ is '__main__':

	import rampedpyrox as rp
