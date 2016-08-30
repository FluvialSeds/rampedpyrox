'''
This module contains helper functions for timedata classes.
'''

#define function to extract variables from 'real_data'
def _rpo_extract_tg(all_data, nT):
	'''
	Extracts time, temperature, and carbon remaining vectors from all_data.
	Called by ``RealData.__init__()``.

	Parameters
	----------
	all_data : str or pd.DataFrame 
		File containing thermogram data, either as a path string or 
		``pd.DataFrame`` instance.

	nT : int 
		The number of time points to use. Defaults to 250.

	Returns
	-------
	t : np.ndarray
		Array of timepoints (in seconds).

	Tau : np.ndarray
		Array of temperature points (in Kelvin).
	
	g : np.ndarray
		Array of fraction of carbon remaining.

	Raises
	------
	ValueError
		If `all_data` is not str or ``pd.DataFrame`` instance.
	
	ValueError
		If `all_data` does not contain "CO2_scaled" and "temp" columns.
	
	ValueError
		If index is not ``pd.DatetimeIndex`` instance.
	'''

	#import all_data as a pd.DataFrame if inputted as a string path and check
	#that it is in the right format
	if isinstance(all_data,str):
		all_data = pd.DataFrame.from_csv(all_data)
	elif not isinstance(all_data,pd.DataFrame):
		raise ValueError('all_data must be pd.DataFrame or path string')

	if 'CO2_scaled' and 'temp' not in all_data.columns:
		raise ValueError('all_data must have "CO2_scaled" and "temp" columns')

	if not isinstance(all_data.index,pd.DatetimeIndex):
		raise ValueError('all_data index must be DatetimeIndex')

	#extract necessary data
	secs = (all_data.index - all_data.index[0]).seconds
	CO2 = all_data.CO2_scaled
	alpha = np.cumsum(CO2)/np.sum(CO2)
	Temp = all_data.temp

	#generate t vector
	t0 = secs[0]; tf = secs[-1]; dt = (tf-t0)/nT
	t = np.linspace(t0,tf,nT+1) + dt/2 #make downsampled points at midpoint
	t = t[:-1] #drop last point since it's beyond tf

	#down-sample g and Tau using interp1d
	fT = interp1d(secs,Temp)
	fg = interp1d(secs,alpha)
	Tau = fT(t)
	g = 1-fg(t)
	
	return t, Tau, g