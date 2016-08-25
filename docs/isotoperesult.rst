Isotope Result
==============



Examples
--------

Calculating the isotope composition of each peak::

	#save string pointing to isotope data
	iso_data = '/path_to_folder_containing_data/isotope_data.csv'

	#create IsotopeResult object using data and ModeledResult object md
	ir = rp.IsotopeResult(iso_data,md)

	#view results
	print(ir.Fm_peak)
	print(ir.d13C_peak)

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
