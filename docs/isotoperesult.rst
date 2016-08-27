Isotope Result
==============
``IsotopeResult`` class instances store the isotope results for each peak
calculated using the measured fraction isotope data and 
``scipy.optimize.least_squares``.


Examples
--------

Calculating the isotope composition of each peak::

	#save string pointing to isotope data
	iso_data = '/path_to_folder_containing_data/isotope_data.csv'

	#create IsotopeResult object using data and ModeledResult object md
	ir = rp.IsotopeResult(iso_data,md)

	#view results
	ir.summary()

Technical documentation
-----------------------

(insert math here)

References
~~~~~~~~~~

(insert references here)

Module Reference
----------------
.. automodule:: rampedpyrox.core.isotoperesult
	:members:
