About rampedpyrox
=================

Rampedpyrox is a Python package for analyzing the results from ramped-temperature
instruments such as RampedPyrox, RockEval, pyrolysis gc (pyGC), thermogravimitry
(TGA), etc. Rampedpyrox deconvolves Gaussian activiation energy peaks within a given
sample using a Distributed Activation Energy Model (DAEM) and calculates the
corresponding stable-carbon and radiocarbon isotope values for each peak.

Rampedpyrox is written by Jordon D. Hemingway (MIT/WHOI Joint Program). When using
rampedpyrox, please cite the following peer-reviewed publications:

(insert papers here once published)


Documentation
=============
The documentation for the latest release is available at:

	http://www.insert_url_here.com

Main Features
=============
* Storing and plotting thermogram data:
	- Real data
	- Inverse model results

* Performing first-order DAEM inverse model:
	- Calculate Laplace Transform
	- User-controlled regularization (smoothing)
		- Plot L-curves

* DAEM deconvolution:
	- Automated or user-defined peak number selection

* Isotope deconvolution:
	- Radiocarbon (Fm) deconvolution
	- Stable-carbon (:math:r'\delta^{13}C') deconvolution


How to Obtain
=============
Source code can be directly downloaded from GitHub:

	http://github.com/FluvialSeds/rampedpyrox

Binaries can be installed through the Python package index

	pip install rampedpyrox

License
=======
This product is licensed under the MIT License.

Bug Reports
===========
This software is still in active deveopment. Please report any bugs directly to me at:

	jordonhemingway@gmail.com