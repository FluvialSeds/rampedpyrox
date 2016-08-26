'''
Thermogram module for storing thermogram data, either RealData (i.e. collected
from an actual instrument) or ModeledData (i.e. results from inverse model).

* TODO: Add summary method.
'''

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.interpolate import interp1d

__docformat__ = 'restructuredtext en'

#exclude Thermogram superclass in __all__
__all__ = ['RealData','ModeledData']

#define function to extract Thermogram variables from 'real_data'
def _extract_tg(all_data, nT):
	'''
	Extracts time, temperature, and carbon remaining vectors from all_data.
	Called by Thermogram during __init__.

	Args:
		all_data (str or pd.DataFrame): File containing thermogram data,
			either as a path string or pandas.DataFrame object.

		nT (int): The number of time points to use.

	Returns:
		t (np.ndarray): Array of timepoints.
		Tau (np.ndarray): Array of temperature points.
		g (np.ndarray): Array of fraction of carbon remaining.

	Raises:
		ValueError: If `all_data` is not str or pd.DataFrame.
		ValueError: If `all_data` does not contain "CO2_scaled" and "temp" columns.
		ValueError: If index is not `DatetimeIndex`.
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
	Base class for thermogram objects.
	This class is intended for subclassing and should never be called directly
	'''

	def __init__(self, t, Tau, g):

		#define public parameters
		self.t = t #seconds
		self.Tau = Tau + 273.15 #Kelvin
		self.g = g #fraction

		self.Taudot_t = np.gradient(Tau)/np.gradient(t) #Kelvin/second
		self.gdot_t = np.gradient(g)/np.gradient(t) #fraction/second
		self.gdot_Tau = self.gdot_t/self.Taudot_t #fraction/Kelvin

	def plot(self, ax=None, xaxis='time'):
		'''
		Plots the thermogram against time or temp.
		Modified by either RealData.plot or ModeledData.plot
		'''

		if xaxis not in ['time','temp']:
			raise ValueError('"xaxis" must be either "time" or "temp"')

		if ax is None:
			_,ax = plt.subplots(1,1)

		if xaxis == 'time':
			tg_line, = ax.plot(self.t,-self.gdot_t)
			ax.set_xlabel('time (seconds)')
			ax.set_ylabel(r'normalized carbon loss rate (fraction/second)')
		else:
			tg_line, = ax.plot(self.Tau,-self.gdot_Tau)
			ax.set_xlabel('Temp. (Kelvin)')
			ax.set_ylabel(r'normalized carbon loss rate (fraction/K)')

		return ax, tg_line

	def summary():
		'''
		Prints a summary of the Thermogram object.
		'''


class RealData(Thermogram):
	__doc__='''
	Class for real data thermograms (e.g. data from RampedPyrox experiment).

	Args:
		all_data (str or pd.DataFrame): File containing thermogram data,
			either as a path string or pandas.DataFrame object.

		nT (int): The number of time points to use. Defaults to 250.

	Returns:
		rd (rp.RealData): RealData object containing thermogram data.

	Raises:
		ValueError: If `all_data` is not str or pd.DataFrame.
		ValueError: If `all_data` does not contain "CO2_scaled" and "temp" columns.
		ValueError: If index is not `DatetimeIndex`.

	Examples:
		Importing data into a thermogram object::
	
			#load modules
			import rampedpyrox as rp

			data = '/path_to_folder_containing_data/data.csv'
			nT = 250 #number of timepoints
			rd = rp.RealData(data,nT=nT)
	'''

	def __init__(self, all_data, nT=250):

		#extract t, Tau, and g from all_data
		t, Tau, g = _extract_tg(all_data, nT)

		#pass variables to Thermogram superclass
		super(RealData,self).__init__(t, Tau, g)

		#define private parameters
		self._nT = nT

	def plot(self, ax=None, xaxis='time'):
		'''
		Plots the true thermogram against time or temp.

		Args:
			ax (None or matplotlib.axis): Axis to plot on. If None, 
				creates an axis object to return. Defaults to None.
			xaxis (str): Sets the x axis units, either 'time' or 'temp'.
				Defaults to 'time'.

		Returns:
			ax (matplotlib.axis): Updated axis with plotted data.

		Raises:
			ValueError: If `xaxis` is not 'time' or 'temp'.

		Examples:
			Plotting thermogram data::

				#load modules
				import matplotlib.pyplot as plt

				fig,ax = plt.subplots(1,1)
				ax = rd.plot(ax=ax,xaxis='time')
		'''

		#plot the thermogram and edit tg_line parameters
		ax,tg_line = super(RealData,self).plot(ax=ax, xaxis=xaxis)
		tg_line.set_linewidth(2)
		tg_line.set_color('k')
		tg_line.set_label('True Thermogram')

		#remove duplicate legend entries (necessary if ModeledData on sam axis)
		handles, labels = ax.get_legend_handles_labels()
		handle_list, label_list = [], []
		for handle, label in zip(handles, labels):
			if label not in label_list:
				handle_list.append(handle)
				label_list.append(label)
		
		ax.legend(handle_list,label_list,loc='best')

		return ax


class ModeledData(Thermogram):
	__doc__='''
	Subclass for thermograms generated using an f(Ea) distribution.

	Args:
		t (np.ndarray): Array of timepoints.
		Tau (np.ndarray): Array of temperature points.
		g (np.ndarray): Array of fraction of carbon remaining.

	Returns:
		md (rp.ModeledData): Modeled object containing thermogram data.

	Raises:
		ValueError: If any of the inputted arrays are not same length.

	Examples:
		Creating a ModeledData object using inversemodel results::
	
			#load modules
			import rampedpyrox as rp

			#t and Tau from RealData object, g_hat and gp from running
			# LaplaceTransform.calc_TG_fwd()
			md = rp.ModeledData(rd.t,rd.Tau,g_hat,gp)
	'''

	def __init__(self, t, Tau, g_hat, gp):

		#ensure equal array lengths
		nT = len(t)

		if len(Tau) != nT:
			raise ValueError('t and Tau arrays must have same length')
		elif len(g_hat) != nT:
			raise ValueError('t and g_hat arrays must have same length')
		elif len(gp) != nT:
			raise ValueError('t and gp arrays must have same length')

		super(ModeledData,self).__init__(t, Tau-273.15, g_hat)

		#calculate tg contribution for each peak
		_,nPeak = np.shape(gp)
		dt_mat = np.gradient(np.outer(t,np.ones(nPeak)),axis=0)
		dTau_mat = np.outer(self.Taudot_t,np.ones(nPeak))
		
		#define public parameters
		self.gp = gp
		self.gpdot_t = np.gradient(gp,axis=0)/dt_mat #second-1
		self.gpdot_Tau = self.gpdot_t/dTau_mat #Kelvin-1

		#define private parameters
		self._nT = nT
		self._nPeak = nPeak

	def plot(self, ax=None, xaxis='time'):
		'''
		Plots the modeled thermogram and individual peaks against time or temp.

		Args:
			ax (None or matplotlib.axis): Axis to plot on. If None, 
				creates an axis object to return. Defaults to None.
			xaxis (str): Sets the x axis units, either 'time' or 'temp'.
				Defaults to time.

		Returns:
			ax (matplotlib.axis): Updated axis with plotted data.

		Raises:
			ValueError: If `xaxis` is not 'time' or 'temp'.

		Examples:
			Plotting thermogram data::

				#load modules
				import matplotlib.pyplot as plt

				fig,ax = plt.subplots(1,1)
				ax = md.plot(ax=ax,xaxis='time')
		'''

		#plot the thermogram and edit tg_line parameters
		ax,tg_line = super(ModeledData,self).plot(ax=ax, xaxis=xaxis)
		tg_line.set_linewidth(2)
		tg_line.set_color('r')
		tg_line.set_label('Modeled Thermogram')

		#plot individual peak contributions
		if xaxis == 'time':
			ax.plot(self.t,-self.gpdot_t,
				'--k',
				linewidth=1,
				label=r'Individual fitted peaks (n=%.0f)' %self._nPeak)
		else:
			ax.plot(self.Tau,-self.gpdot_Tau,
				'--k',
				linewidth=1,
				label=r'Individual fitted peaks (n=%.0f)' %self._nPeak)

		#remove duplicate legend entries
		handles, labels = ax.get_legend_handles_labels()
		handle_list, label_list = [], []
		for handle, label in zip(handles, labels):
			if label not in label_list:
				handle_list.append(handle)
				label_list.append(label)
		
		ax.legend(handle_list,label_list,loc='best')

		return ax
