'''
This module contains helper functions for generating rampedpyrox summary tables.
'''

import numpy as np
import pandas as pd


#define function to calculate timedata peak info and store
def _timedata_peak_info(timedata):
	'''
	Calculates the ``TimeData`` instance peak info and stores as a 
	``pd.DataFrame``.

	Parameters
	----------
	timedata : rp.TimeData
		TimeData instance containing peaks to be summarized.

	Returns
	-------
	df : pd.DataFrame
		DataFrame instance of resulting peak info.

	Raises
	------
	AttributeError
		If TimeData instance does not contain necessary attributes (i.e. if it
		does not have inputted model-estimated data).
	'''

	#raise exception if timedata doesn't contain peaks
	if not hasattr(timedata, 'gam'):
		raise AttributeError((
			"TimeData instance contains no model-fitted data! Run forward"
			"model before trying to summarize peaks."))

	#set pandas display options
	pd.set_option('precision', 2)

	#calculate peak indices
	i = np.argmax(-self.dcmptdt, axis=0)

	#extract info at peaks
	t_max = self.t[i]
	T_max = self.T[i]
	height_t = -self.dcmptdt[i]
	height_T = -self.dcmptdT[i]
	rel_area = np.max(self.cmpt, axis=0)

	peak_info = np.column_stack((t_max, T_max, height_t, height_T, rel_area))
	peak_info = pd.DataFrame(peak_info, 
		columns = ['t_max (s)', 'T_max (K)', 'max_rate (frac/s)', \
			'max_rate (frac/K)','rel. area'],
		index = np.arange(1, self.nPeak + 1))




