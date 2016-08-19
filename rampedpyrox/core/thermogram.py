import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d


#define function to extract Thermogram variables from 'real_data'
def _extract_tg(all_data, nT):
	'''
	Extracts time, temperature, and carbon remaining vectors from all_data.
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


class Thermogram(object):
	'''
	Base class for thermogram objects, intended for subclassing.
	'''

	def __init__(self, t, Tau, g):
		'''
		Initializes the Thermogram object.
		'''

		#define public parameters
		self.t = t #seconds
		self.Tau = Tau + 273.15 #Kelvin
		self.g = g #fraction

		self.Taudot_t = np.gradient(Tau)/np.gradient(t) #Kelvin/second
		self.gdot_t = np.gradient(g)/np.gradient(t) #second-1
		self.gdot_Tau = self.gdot_t/self.Taudot_t #Kelvin-1

	def plot(self, ax=None, xaxis='time'):
		'''
		Plots the thermogram against time or temp.
		'''

		if xaxis not in ['time','temp']:
			raise ValueError('"xaxis" must be either "time" or "temp"')

		if ax is None:
			_,ax = plt.subplots(1,1)

		if xaxis == 'time':
			ax.plot(self.t,-self.gdot_t)
			ax.set_xlabel('time (seconds)')
			ax.set_ylabel(r'rate constant ($s^{-1}$)')
		else:
			ax.plot(self.Tau,-self.gdot_Tau)
			ax.set_xlabel('Temp. (Kelvin)')
			ax.set_ylabel(r'rate constant ($K^{-1}$)')

	def summary():
		'''
		Prints a summary of the Thermogram object.
		'''


class RealData(Thermogram):
	'''
	Subclass for real data thermograms (i.e. actual dirt burner data).
	'''

	def __init__(self, all_data, nT=250):
		'''
		Initializes the RealData object.
		'''

		#extract t, Tau, and g from all_data
		t, Tau, g = _extract_tg(all_data, nT)

		#pass variables to Thermogram superclass
		super(RealData,self).__init__(t, Tau, g)

		#define private parameters
		self._nT = nT


class ModeledData(Thermogram):
	'''
	Subclass for thermograms generated using an f(Ea) distribution.
	'''

	def __init__():
		'''
		Initializes the ModeledData object.
		'''

		super(ModeledData,self).__init__(t, tau, g)

