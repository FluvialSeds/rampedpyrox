'''
This module contains helper functions for plotting rampedpyrox data.
'''


from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_plot_dicts', '_rem_dup_leg']

import numpy as np

#define function to pull plotting dicts
def _plot_dicts(case, td):
	'''
	Function to access different plotting dicts.

	Parameters
	----------
	case : str
		The case that defines the dict to pull.
		Acceptable strings:

			'rpo_rd', \n
			'rpo_labs', \n
			'rpo_md'

	td : TimeData or subclass
		``rp.TimeData`` instance containing the data to plot.

	Returns
	-------
	pl_dict : dict
		The resulting dictionary containing plotting info.
	'''

	if case == 'rpo_labs':
		#create a nested dict to keep track of axis labels
		pl_dict = {'time': 
						{'fraction' : ('time (s)', 'g (unitless)'),
						'rate' : ('time (s)', r'fraction/time $(s^{-1})$')
						},
					'temp' : 
						{'fraction' : ('temp (K)', 'g (unitless)'),
						'rate' : ('temp (K)', r'fraction/temp $(K^{-1})$')}
						}
	
	elif case == 'rpo_md':
		#create a nested dict to keep track of cases of modeled data
		pl_dict = {'time': 
						{'fraction' : (td.t, td.ghat),
						'rate' : (td.t, -td.dghatdt)
						},
					'temp': 
						{'fraction' : (td.T, td.ghat),
						'rate' : (td.T, -td.dghatdT)}
						}

	elif case == 'rpo_rd':
		#create a nested dict to keep track of cases for real data
		pl_dict = {'time': 
						{'fraction' : (td.t, td.g),
						'rate' : (td.t, -td.dgdt)
						},
					'temp': 
						{'fraction' : (td.T, td.g),
						'rate' : (td.T, -td.dgdT)}
						}

	return pl_dict

#define function to pull plotting dicts
def _plot_dicts_iso(case, ri):
	'''
	Function to access different plotting dicts.

	Parameters
	----------
	case : str
		The case that defines the dict to pull.
		Acceptable strings:

			'rpo_rd', \n
			'rpo_labs', \n
			'rpo_md'

	ri : Results or subclass
		``rp.Results`` instance containing the data to plot.

	Returns
	-------
	pl_dict : dict
		The resulting dictionary containing plotting info.
	'''

	if case == 'rpo_iso_labs':
		#create a nested dict to keep track of isotope result axis labels
		pl_dict = {'E': 
						{'p0E' : (r'E (kJ $mol^{-1}$)', 
							r'p(E) (unitless)'),
						'Fm' : (r'E (kJ $mol^{-1}$)', r'Fm'),
						'd13C' : (r'E (kJ $mol^{-1}$)', 
							r'$\delta^{13}C$ (VPDB)')}
						}

	elif case == 'iso_corr':
		#create a nested dict to keep track of cases of scatter
		pl_dict = {'E': 
						{'Fm_corr' : (ri.E_frac, ri.Fm_corr, 
							ri.E_frac_std, ri.Fm_corr_std),
						'd13C_corr' : (ri.E_frac, ri.d13C_corr, 
							ri.E_frac_std, ri.d13C_corr_std)}
						}

	elif case == 'iso_raw':
		#create a nested dict to keep track of cases of scatter
		pl_dict = {'E': 
						{'Fm_raw' : (ri.E_frac, ri.Fm_raw, 
							ri.E_frac_std, ri.Fm_raw_std),
						'd13C_raw' : (ri.E_frac, ri.d13C_raw, 
							ri.E_frac_std, ri.d13C_raw_std)}
						}

	return pl_dict

#define function to remove duplicate legend entries
def _rem_dup_leg(ax):
	'''
	Removes duplicate legend entries.

	Parameters
	----------
	ax : plt.axishandle
		Axis handle containing entries to remove.

	Returns
	-------
	han_list : list
		List of axis handles.

	lab_list : list
		List of axis handle labels.
	'''
	han, lab = ax.get_legend_handles_labels()
	han_list, lab_list = [], []
	
	for h, l in zip(han, lab):
		
		if l not in lab_list:
		
			han_list.append(h)
			lab_list.append(l)

	return han_list, lab_list
