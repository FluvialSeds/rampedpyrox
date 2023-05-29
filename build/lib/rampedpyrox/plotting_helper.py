'''
This module contains helper functions for plotting rampedpyrox data.
'''


from __future__ import(
	division,
	print_function,
	)

__docformat__ = 'restructuredtext en'
__all__ = ['_bd_plot_bge',
			'_plot_dicts', 
			'_plot_dicts_iso',
			'_rem_dup_leg',
			]

import numpy as np

#define function to plot carbon flux overlaid by BGE
def _bd_plot_bge(
	t_elapsed,
	bge,
	bge_err = None,
	ax = None,
	ymin = 0.0,
	ymax = 1.0):
	'''
	Function to plot the carbon flux (in ugC min-1 L-1) overlaid by bacterial
	growth efficiency for each time bin.

	Parameters
	----------
	t_elapsed : pd.Series
		Series containing the time elapsed (in minutes), with pd.DatetimeIndex
		as index.

	bge : pd.Series
		Series containing calculated BGE values, reported at the final
		timepoint for a given value.

	bge_err : None or pd.Series
		Series containing uncertainties for BGE values, reported at the final
		timepoint for a given value. If `None`, no uncertainty is plotted.
		Defaults to `None`.

	ax : None or matplotlib.axis
		Axis to plot BGE data on. If `None`, automatically creates a
		``matplotlip.axis`` instance to return. Defaults to `None`.

	ymin : float
		Minimum y value for BGE axis. Defaults to `0.0`.

	ymax : float
		Maximum y value for BGE axis. Defaults to `1.0`.

	Returns
	-------
	ax : matplotlib.axis
		Axis containing BGE data
	'''
	#create axis if necessary and label
	if ax is None:
		_, ax = plt.subplots(1, 1)

	#find t_elapsed values for each entry in bge
	bge_inds = bge.index
	bge_times = t_elapsed[bge_inds]

	#loop through each time range and plot BGE
	for i, ind in enumerate(bge_inds[1:]):
		#find bounding time points
		t0 = t_elapsed[bge_inds[i]]
		tf = t_elapsed[ind]
		b = bge[i+1]

		#plot results
		ax.plot(
			[t0, tf],
			[b, b],
			c = 'k',
			linewidth = 2
			)

		#include uncertainty as a shaded box
		if bge_err is not None:

			berr = bge_err[i+1]

			ax.fill_between(
				[t0, tf],
				b - berr,
				b + berr,
				alpha = 0.5,
				color = 'k',
				linewidth = 0
				)

	#set limits and label
	ax.set_ylim([ymin, ymax])
	ax.set_ylabel('Bacterial Growth Efficiency (BGE)')

	return ax

#define function to pull plotting dicts
def _plot_dicts(case, td):
	'''
	Function to access different plotting dicts.

	Parameters
	----------
	case : str
		The case that defines the dict to pull.
		Acceptable strings:

			'bd_labs', \n
			'bd_md', \n
			'bd_rd', \n
			'rpo_labs', \n
			'rpo_md', \n
			'rpo_rd'
			
	td : TimeData or subclass
		``rp.TimeData`` instance containing the data to plot.

	Returns
	-------
	pl_dict : dict
		The resulting dictionary containing plotting info.
	'''

	if case == 'bd_labs':
		#create a dict to keep track of axis labels
		pl_dict = {'secs': 
						{'fraction' : ('time (s)', 'g (unitless)'),
						'rate' : ('time (s)', r'fraction/time $(s^{-1})$')
						},
					'mins' : 
						{'fraction' : ('time (min)', 'g (unitless)'),
						'rate' : ('time (min)', r'fraction/temp $(K^{-1})$')
						},
					'hours' : 
						{'fraction' : ('time (hr)', 'g (unitless)'),
						'rate' : ('time (hr)', r'fraction/temp $(K^{-1})$')
						},
					'days' : 
						{'fraction' : ('time (d)', 'g (unitless)'),
						'rate' : ('time (d)', r'fraction/temp $(K^{-1})$')}
						}

	elif case == 'bd_md':
		#create a dict to keep track of cases of modeled data
		pl_dict = {'secs': 
						{'fraction' : (td.t, td.ghat),
						'rate' : (td.t, -td.dghatdt)
						},
					'mins': 
						{'fraction' : (td.t / 60, td.ghat),
						'rate' : (td.t / 60, -td.dghatdT)
						},
					'hours': 
						{'fraction' : (td.t / (60*60), td.ghat),
						'rate' : (td.t / (60*60), -td.dghatdT)
						},
					'days': 
						{'fraction' : (td.t / (60*60*24), td.ghat),
						'rate' : (td.t / (60*60*24), -td.dghatdT)}
						}

	elif case == 'bd_rd':
		#create a dict to keep track of cases for real data
		pl_dict = {'secs': 
						{'fraction' : (td.t, td.g),
						'rate' : (td.t, -td.dgdt)
						},
					'mins': 
						{'fraction' : (td.t / 60, td.g),
						'rate' : (td.t / 60, -td.dgdT)
						},
					'hours': 
						{'fraction' : (td.t / (60*60), td.g),
						'rate' : (td.t / (60*60), -td.dgdT)
						},
					'days': 
						{'fraction' : (td.t / (60*60*24), td.g),
						'rate' : (td.t / (60*60*24), -td.dgdT)}
						}

	elif case == 'rpo_labs':
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
							r'p(0,E)'),
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
