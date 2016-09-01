## testing descriptor class for inputting lengh nt arrays

import numpy as np

from collections import Sequence


class rparray(np.ndarray):
	__doc__='''
	Subclass of np.ndarray that forces a particular length. Used for
	ensuring all arrays in ``rp.TimeData`` instance have length `nt`, 
	arrays in ``rp.RateData`` instance have length `nk`, etc.

	Parameters
	----------
	data : scalar or array-like
		The data to be inputted into ``rparray``. If scalar, data gets
		converted to float and projected onto an array of length `n`.
		Data can be 1d or 2d array-like.

	n : scalar
		The forced length of the array (first dimension).

	Keyword Arguments
	-----------------
	sig_figs : int
		Number of significant figures to retain. Defaults to 6.

	Additional ``np.array`` keyword arguments

	Raises
	------
	TypeError
		If `data` is not int, float, or array-like

	ValueError
		If length of `data` is not `n`.

	Examples
	--------
	Generating an ``rparray`` instance of length, `n`::

		#import modules
		import numpy as np
		import rampedpyrox as rp

		data = np.ones([n,n])

		array = rparray(data, n,
			sig_figs = 1)

	Calculating the derivative with respect to a second ``rparray`` instance,
	`array_2`::

		deriv = array.derivatize(array_2, 
			sig_figs = 1)

	References
	----------
	van der Walt et al. (2011). *The NumPy Array: A Structure for Efficient 
		Numerical Computation, *Computing in Science & Engineering*, **13**, 
		22-30, DOI:10.1109/MCSE.2011.37

	Attributes
	----------
	n : int
		The forced length of the array.

	``np.ndarray`` attributes

	'''

	def __new__(cls, data, n, copy=True, order=None, subok=True, sig_figs=6):

		#assert that n and sig_figs are int
		n = int(n); sig_figs = int(sig_figs)

		#assert data is in the right form
		if isinstance(data, (int, float)):
			data = data*np.ones(n)
		
		elif isinstance(data, Sequence) or hasattr(data, '__array__'):
			
			if isinstance(data, str):
				raise TypeError('data cannot be a string')

			elif len(data) != n:
				raise ValueError(
					'Cannot create rparray of length %r if n = %r' \
					% (len(data), n))

		else:
			raise TypeError('data must be scalar or array-like')
			

		#(re-)construct an ndarray using the array constructor
		array = np.array(data,
			dtype=float, #ensure float
			copy=copy, 
			order=order, 
			subok=subok, 
			ndmin=1, #ensure at least 1d
			)

		#round array to sig_figs
		array = array.round(decimals=sig_figs)

		#cast ndarray instance onto rparray
		obj = array.view(cls)

		#add additional attribute, n
		obj.n = n

		return obj

	def __array_finalize__(self, obj):

		#return if constructing directly
		if obj is None: return

		#set default n value for construction
		self.n = getattr(obj, 'n', None)

	def derivatize(self, denom, sig_figs=6):
		'''
		Method to calculate derivative of an ``rdarray`` instance with respect
		to the inputted denominator, `denom`.

		Parameters
		----------
		denom : scalar or array-like
			The denominator of the numerical derivative function.

		Keyword Arguments
		-----------------
		sig_figs : int
			Number of significant figures to retain. Defaults to 6.
		
		Returns
		-------
		derivative : rparray
			An ``rparray`` instance of the derivative.

		See Also
		--------
		np.gradient
			The method used to calculate derivatives

		rp.derivatize
			Package method of ``derivatize``.

		Notes
		-----
		This method uses the ``np.gradient`` method to calculate derivatives.

		If `denom` is a scalar, resulting array will be all ``np.inf``.
		
		If both `self` and `denom` are scalars, resulting array will be all
		``np.nan``.

		If either `denom` or `self` are 1d and the other is 2d, derivative
		will be calculated column-wise. If both are 2d, each column will
		be derivatized separately.
		'''

		#make sure denom is the right length and type
		n = self.n
		denom = rparray(denom, n)

		#calculate separately for each dimensionality case
		if self.ndim == denom.ndim == 1:
			dndd = np.gradient(self)/np.gradient(denom)

		elif self.ndim == denom.ndim == 2:
			dndd = np.gradient(self)[0]/np.gradient(denom)[0]

		#note recursive list comprehension when dimensions are different
		elif self.ndim == 2 and denom.ndim == 1:
			col_der = [col.derivatize(denom) for col in self.T]
			dndd = np.column_stack(col_der)

		elif self.ndim == 1 and denom.ndim == 2:
			col_der = [self.derivatize(col) for col in denom.T]
			dndd = np.column_stack(col_der)

		return rparray(dndd, n, sig_figs=sig_figs)















