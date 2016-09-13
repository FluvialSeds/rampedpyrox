.. PB-60 pyrolysis comparison
.. Sarah's matrix effects

Comprehensive Walkthrough
=========================
The following examples should form a comprehensive walkthough of downloading the package, getting thermogram data into the right form for importing, running the DAEM inverse model, peak-fitting the activation energy (Ea) probability density function, determining the isotope composition of each Ea Gaussian peak, and performing Monte Carlo isotope uncertainty estimates.


Quick guide
-----------

Basic runthrough::

	#import modules
	import matplotlib.pyplot as plt
	import numpy as np
	import pandas as pd
	import rampedpyrox as rp

	#generate string to data
	tg_data = '/folder_containing_data/tg_data.csv'
	iso_data = '/folder_containing_data/iso_data.csv'

	#make the thermogram instance
	tg = rp.RpoThermogram.from_csv(
		tg_data,
		bl_subtract = True,
		nt = 250)

	#generate the DAEM
	daem = rp.Daem.from_timedata(
		tg,
		log10k0 = 10, #assume a constant value of 10
		Ea_max = 350,
		Ea_min = 50,
		nEa = 400)

	#run the inverse model to generate energy complex
	ec = rp.EnergyComplex.inverse_model(
		daem, 
		tg,
		combined = [(1,2), (6, 7)],
		nPeaks = 'auto',
		omega = 3,
		peak_shape = 'Gaussian',
		thres = 0.02)

	#forward-model back onto the thermogram
	tg.forward_model(daem, ec)

	#make the isotope results instance
	ri = rp.RpoIsotopes.from_csv(
		iso_data,
		blk_corr = True, #uses values for NOSAMS instrument
		mass_err = 0.01)

	#fit the component isotope values and uncertainty
	ri.fit(
		daem, 
		ec, 
		tg,
		DEa = 0.0018, #uses values for NOSAMS instrument
		nIter = 10000)


Downloading the package
-----------------------

Using the ``pip`` package manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
``rampedpyrox`` and the associated dependent packages can be downloaded directly from the command line using ``pip``::

	$ pip install rampedpyrox

You can check that your installed version is up to date with the latest release by doing::

	$ pip freeze

**This option will become available once the peer-reviewed manuscripts accompanying this package have been published.**


Downloading from source
~~~~~~~~~~~~~~~~~~~~~~~
Alternatively, ``rampedpyrox`` source code can be downloaded directly from `my github repo <http://github.com/FluvialSeds/rampedpyrox>`_. Or, if you have git installed::

	$ git clone git://github.com/FluvialSeds/rampedpyrox.git

And keep up-to-date with the latest version by doing::

	$ git pull

from within the rampedpyrox directory.

**This github repo is currently private until the peer-reviewed manuscripts accompanything this package have been published.** Please `contact me directly <jhemingway@whoi.edu>`_ for access.

Dependencies
~~~~~~~~~~~~
The following packages are required to run ``rampedpyrox``:

* `python <http://www.python.org>`_ >= 2.7, including Python 3.x

* `matplotlib <http://matplotlib.org>`_ >= 1.5.2

* `numpy <http://www.numpy.org>`_ >= 1.11.1

* `pandas <http://pandas.pydata.org>`_ >= 0.18.1

* `scipy <http://www.scipy.org>`_ >= 0.18.0

If downloading using ``pip``, these dependencies (except python) are installed
automatically.

Optional Dependencies
~~~~~~~~~~~~~~~~~~~~~
The following packages are not required but are highly recommended:

* `ipython <http://www.ipython.org>`_ >= 4.1.1

Additionally, if you are new to the Python environment or programming using the command line, consider using a Python integrated development environment (IDE) such as:

* `wingware <http://wingware.com>`_

* `Enthought Canopy <https://store.enthought.com/downloads/#default>`_

* `Anaconda <https://www.continuum.io/downloads>`_

* `Spyder <https://github.com/spyder-ide/spyder>`_

Python IDEs provide a "MATLAB-like" environment as well as package management. This option should look familiar for users coming from a MATLAB or RStudio background.

Getting data in the right format
--------------------------------

Importing thermogram data
~~~~~~~~~~~~~~~~~~~~~~~~~
For thermogram data, this package requires that the file is in `.csv` format, that the first column is `date_time` index in an **hh:mm:ss AM/PM** format, and that the file contains 'CO2_scaled' and 'temp' columns [1]_. For example:

+-------------+------------+--------------+
|  date_time  |    temp    |  CO2_scaled  |
+=============+============+==============+
|10:24:20 AM  |  100.05025 |    4.6       |
+-------------+------------+--------------+
|10:24:21 AM  |  100.09912 |    5.3       |
+-------------+------------+--------------+
|10:24:22 AM  |  100.11413 |    5.1       |
+-------------+------------+--------------+
|10:24:23 AM  |  100.22759 |    4.9       |
+-------------+------------+--------------+

Once the file is in this format, generate a string pointing to it in python 
like this::

	#create string of path to data
	all_data = '/path_to_folder_containing_data/all_data.csv'

Importing isotope data
~~~~~~~~~~~~~~~~~~~~~~
If you are importing isotope data, this package requires that the file is in `.csv` format and that the first two rows correspond to the starting time of the experiment and the initial trapping time of fraction 1, respectively. Additionally, the file must contain a 'fraction' column and isotope/mass columns must have `ug_frac`, `d13C`, `d13C_std`, `Fm`, and `Fm_std` headers [2]_. For example:

+-------------+----------+---------+--------+----------+--------+----------+
|  date_time  | fraction | ug_frac |  d13C  | d13C_std |   Fm   |  Fm_std  |
+=============+==========+=========+========+==========+========+==========+
|10:24:20 AM  |    -1    |    0    |    0   |    0     |    0   |     0    |
+-------------+----------+---------+--------+----------+--------+----------+
|10:45:10 AM  |     0    |    0    |    0   |    0     |    0   |     0    |
+-------------+----------+---------+--------+----------+--------+----------+
|11:32:55 AM  |     1    |  69.05  | -30.5  |   0.1    | 0.8874 |  0.0034  |
+-------------+----------+---------+--------+----------+--------+----------+
|11:58:23 AM  |     2    | 105.81  | -29.0  |   0.1    | 0.7945 |  0.0022  |
+-------------+----------+---------+--------+----------+--------+----------+

Here, the `ug_frac` column is composed of manometrically determined masses rather than those determined by the infrared gas analyzer (IRGA, *i.e.* photometric). As such, the mass RMSE value determined by the fitting procedure (see `Determining component isotope composition`_ below) is a metric of the discrepancy between photometric and manometric mass measurements in addition to that between the peak-fitted and true thermograms.

**Important:** The `date_time` value for fraction '-1' must be the same as the `date_time` value for the first row in the `all_data` thermogram file **and** the value for fraction '0' must the initial time when trapping for fraction 1 began.

Once the file is in this format, generate a string pointing to it in python like this::

	#create string of path to data
	sum_data = '/path_to_folder_containing_data/sum_data.csv'

Making a TimeData instance (the Thermogram)
-------------------------------------------
Once the `all_data` string been defined, you are ready to import the package and generate an ``rp.RpoThermogram`` instance containing the thermogram data. ``rp.RpoThermogram`` is a subclass of ``rp.TimeData`` -- broadly speaking, this handles any object that contains measured time-series data. It is important to keep in mind that your thermogram will be down-sampled to `nt` points in order to smooth out high-frequency noise and to keep Laplace transform matrices to a manageable size for inversion (see `Setting-up the model`_ below). Additionally, because the inversion model is sensitive to boundary conditions at the beginning and end of the run (see `Deconvolving rate data into peaks`_ below), there is an option when generating the thermogram instance to ensure that the baseline has been subtracted, as well as options for inputting measurement uncertainty (time data uncertainty is currently unused as of v.0.0.2)::

	#load modules
	import rampedpyrox as rp

	#number of timepoints to be used in down-sampled thermogram
	nt = 250

	tg = rp.RpoThermogram.from_csv(
		data,
		bl_subtract = True, #subtract baseline
		nt = nt,
		ppm_CO2_err = 5, #IRGA measurement uncertainty
		T_err = 1) #thermocouple uncertainty

Plot the thermogram and the fraction of carbon remaining against temperature [3]_ or time::

	#load modules
	import matplotlib.pyplot as plt

	#make a figure
	fig, ax = plt.subplots(2, 2, 
		figsize = (8,8), 
		sharex = 'col')

	#plot results
	ax[0, 0] = tg.plot(
		ax = ax[0, 0], 
		xaxis = 'time',
		yaxis = 'rate')

	ax[0, 1] = tg.plot(
		ax = ax[0, 1], 
		xaxis = 'temp',
		yaxis = 'rate')

	ax[1, 0] = tg.plot(
		ax = ax[1, 0], 
		xaxis = 'time',
		yaxis = 'fraction')

	ax[1, 1] = tg.plot(
		ax = ax[1, 1], 
		xaxis = 'temp',
		yaxis = 'fraction')

	#adjust the axes
	ax[0, 0].set_ylim([0, 0.00032])
	ax[0, 1].set_ylim([0, 0.0035])
	ax[1, 1].set_xlim([375, 1200])

	plt.tight_layout()

Resulting plots look like this:

|realdata|

Setting-up the model
--------------------

The Laplace transform
~~~~~~~~~~~~~~~~~~~~~
Once the ``rp.RpoThermogram`` instance has been created, you are ready to run the inversion model and generate a regularized and discretized probability density function (pdf) of the rate/activation energy distribution, `f` [4]_. For non-isothermal thermogram data, this is done using a first-order Distributed Activation Energy Model (DAEM) [5]_ by generating an ``rp.Daem`` instance containing the proper Laplace Transform matrix, `A`, to translate between time and activation energy space. This matrix contains all the assumptions that go into building the DAEM inverse model as well as all of the information pertaining to experimental conditions (*e.g.* ramp rate) [6]_. Importantly, the `A` matrix does not contain any information about the sample itself -- it is simply the model "design" -- and a single ``rp.Daem`` instance can be used for multiple samples provided they were analyzed under identical experimental conditions.

One critical user input for the DAEM is the Arrhenius pre-exponential factor, `k0` (inputted here in log\ :sub:`10`\  form). Because there is much discussion in the literature over the constancy and best choice of this parameter (the so-called 'kinetic compensation effect' or KCE [7]_), this package allows `log10k0` to be inputted as a constant, an array, or a function of Ea.

For convenience, you can create any model directly from either time data or rate data, rather than manually inputting time, temperature, and rate vectors. Here, I create a DAEM using the thermogram defined above and allow Ea to range from 50 to 400 kJ/mol::

	#define log10k0, assume constant value of 10
	log10k0 = 10

	#define Ea range (in kJ/mol)
	Ea_min = 50
	Ea_max = 400
	nEa = 400 #number of points in the vector

	#create the DAEM instance
	daem = rp.Daem.from_timedata(
		tg,
		log10k0 = log10k0,
		Ea_max = Ea_max,
		Ea_min = Ea_min,
		nEa = nEa)

Regularizing the inversion
~~~~~~~~~~~~~~~~~~~~~~~~~~
Once the model has been created, you must tell the package how much to 'smooth' the resulting f(Ea) distribution. This is done by choosing an `omega` value to be used as a smoothness weighting factor for Tikhonov regularization [8]_. Higher values of `omega` increase how much emphasis is placed on minimizing changes in the first derivative at the expense of a better fit to the measured data, which includes analytical uncertainty -- practically speaking, regularization aims to "fit the data while ignoring the noise." This package can calculate a best-fit `omega` value using the L-curve method [6]_ by doing.

Here, I calculate and plot L curve for the thermogram and model defined above::

	#make a figure
	fig,ax = plt.subplots(1, 1,
		figsize = (5, 5))

	om_best, ax = daem.calc_L_curve(rd, ax = ax)

	plt.tight_layout()

Resulting L-curve plot looks like this, here with a calculated best-fit omega
value of 0.448:

|lcurve|

**Important:** Best-fit `omega` values generated by the L-curve method typically under-regularize f(Ea) when used for Ramped PyrOx isotope deconvolution. That is, f(Ea) distributions will contain more peaks than can be resolved using the ~5-7 CO\ :sub:`2`\  fractions typically collected during a Ramped PyrOx run. This can be partially addressed by combining peaks when deconvolving the rate data using the ``comine`` flag (see `Deconvolving rate data into peaks`_ below) [9]_.  Alternatively, you can increase the `omega` value (a value of ~1-5 will result in ~5-6 Gaussian peaks for most samples).


Making a RateData instance (the inversion results)
--------------------------------------------------
After creating the ``rp.Daem`` instance and deciding on a value for `omega`, you are ready to invert the thermogram and generate an Activation Energy Complex (EC). An EC is a subclass of the more general ``rp.RateData`` instance which, broadly speaking, contains all rate and/or activation energy information. That is, the EC contains an estimate of the underlying Ea distribution, f(Ea), that is intrinsic to a particular sample for a particular degradation experiment type (*e.g.* combustion, pyrolysis, enzymatic degradation, etc.). A fundamental facet of this model is the realization that each distribution is composed of a sum of individual peaks, each with unique activation energies, isotope, and molecular compositions.

Deconvolving rate data into peaks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The next step is to deconvolve the inverse-modeled rate data distribution into individual peaks. It is important to realize that, until now, the model has made no assumptions about the shape of f(Ea) or the individual peaks that is comprises. The fact that the regularized f(Ea) resembles a sum of Gaussian peaks appears to be a fundamental property of complex organic carbon mixtures, as has been discussed before [10]_. 

Generating the ``up.EnergyComplex`` instance using the inverse model will automatically deconvolve f(Ea) into a sum of peaks. Here we can add user-inputted information for performing the deconvolution: 

* the ``omega`` value used for regulariation (see `Regularizing the inversion`_ above).

* the shape of the underlying peaks, ``peak_shape`` (only 'Gaussian' is supported as of v.0.0.2).

* the number of peaks to retain by the model (``nPeaks``), either as an integer or 'auto'. Peaks are automatically detected according to the curvature of f(Ea) -- each unique concave-down region is assumed to contain a single peak. Peaks below the relative threshold cutoff, ``thres``, are ignored (*e.g.* for the below example, anything below 2% of the largest peak). A higher value of `omega` will lead to less detectable peaks.

* which, if any, peaks to combine (``combined``). Sometimes the maximum number of peaks whose unique isotope composition can be determined is limited due to low sample resolution (see `Determining component isotope composition`_ below). Typically, this occurs when two or more peaks reside exclusively within a single isotope measurement region. (*e.g.* for the below example, the pairs 1 & 2 and 6 & 7 are combined) [9]_.

Here I create two energy complexes, one with `omega` set to 'auto' and the other with `omega` set to 3, and perform the deconvolution by inverse modeling the above thermogram::

	ec_auto = rp.EnergyComplex.inverse_model(
		daem, 
		tg,
		combined = None,
		nPeaks = 'auto',
		omega = 'auto',
		peak_shape = 'Gaussian',
		thres = 0.02)

	ec_3 = rp.EnergyComplex.inverse_model(
		daem, 
		tg,
		combined = [(1,2), (6, 7)],
		nPeaks = 'auto',
		omega = 3,
		peak_shape = 'Gaussian',
		thres = 0.02)

Plot the resulting deconvolved energy complex::

	#make a figure
	fig,ax = plt.subplots(1, 2, 
		figsize = (8,5),
		sharey = True)

	#plot results
	ax[0] = ec_auto.plot(ax = ax[0])
	ax[1] = ec_3.plot(ax = ax[1])

	ax[0].set_title("omega = 'auto'")
	ax[1].set_title("omega = 3")

	ax[0].set_ylim([0, 0.022])
	plt.tight_layout()

Resulting plots are shown side-by-side:

|phis|

Note that the number of peaks reported in the legend is before the ``combined`` flag has been implemented. The first and last 2 peaks are dhown combined in the `omega = 3` plot.

A summary of the peaks can be printed with the ``peak_info`` attribute and saved to a `.csv` file::

	ec_3.peak_info
	ec_3.peak_info.to_csv('EC_peak_info_file.csv')

This will print a table similar to:

+-------------------------------------------------------------+
|Information for each deconvolved peak:                       |
+=====+=============+================+==========+=============+
|     | mu (kJ/mol) | sigma (kJ/mol) |  height  |  rel. area  |
+-----+-------------+----------------+----------+-------------+
|  1  |  134.36     |   7.75         | 3.87e-3  |  0.08       |
+-----+-------------+----------------+----------+-------------+
|  2  |  151.81     |   8.62         | 9.95e-3  |  0.21       |
+-----+-------------+----------------+----------+-------------+
|  3  |  175.25     |   9.46         | 6.99e-3  |  0.17       |
+-----+-------------+----------------+----------+-------------+
|  4  |  202.60     |   9.96         | 6.43e-3  |  0.16       |
+-----+-------------+----------------+----------+-------------+
|  5  |  228.73     |   8.29         | 1.54e-3  |  0.32       |
+-----+-------------+----------------+----------+-------------+
|  6  |  262.32     |   6.18         | 2.41e-3  |  0.04       |
+-----+-------------+----------------+----------+-------------+
|  7  |  282.85     |   7.89         | 1.32e-3  |  0.03       |
+-----+-------------+----------------+----------+-------------+

Additionally, the deconvolution RMSE can be printed as a metric of the quality of the fit::

	print(ec_3.rmse)

Forward modeling the estimated thermogram
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
Once the ``rp.EnergyComplex`` instance has been created, you can forward-model the predicted thermogram and compare with measured data using the ``forward_model`` method of any ``rp.TimeData`` instance. Here, I'll forward model the results from the `omega = 3` energy complex::

	tg.forward_model(daem, ec_3)

The thermogram is now updated with modeled data and can be plotted::
	
	#make a figure
	fig, ax = plt.subplots(2, 2, 
		figsize = (8,8), 
		sharex = 'col')

	#plot results
	ax[0, 0] = tg.plot(
		ax = ax[0, 0], 
		xaxis = 'time',
		yaxis = 'rate')

	ax[0, 1] = tg.plot(
		ax = ax[0, 1], 
		xaxis = 'temp',
		yaxis = 'rate')

	ax[1, 0] = tg.plot(
		ax = ax[1, 0], 
		xaxis = 'time',
		yaxis = 'fraction')

	ax[1, 1] = tg.plot(
		ax = ax[1, 1], 
		xaxis = 'temp',
		yaxis = 'fraction')

	#adjust the axes
	ax[0, 0].set_ylim([0, 0.00032])
	ax[0, 1].set_ylim([0, 0.0035])
	ax[1, 1].set_xlim([375, 1200])

	plt.tight_layout()

Resulting plot looks like this:

|modeleddata|

Similar to ``rp.EnergyComplex``, you can print and save a summary of the components::

	tg.cmpt_info
	tg.peak_info.to_csv('tg_peak_info_file.csv')

Which will print a table similar to:

+---------------------------------------------------------------------------------+
|Information for each deconvolved component:                                      |
+=====+===========+===========+===================+===================+===========+
|     | t max (s) | T max (K) | max rate (frac/s) | max rate (frac/K) | rel. area |
+-----+-----------+-----------+-------------------+-------------------+-----------+
|  1  | 3200.73   | 622.70    | 1.82e-4           | 2.25e-3           | 0.29      |
+-----+-----------+-----------+-------------------+-------------------+-----------+
|  2  | 4481.02   | 728.72    | 1.17e-4           | 1.36e-3           | 0.17      |
+-----+-----------+-----------+-------------------+-------------------+-----------+
|  3  | 5717.17   | 832.23    | 1.05e-4           | 1.31e-3           | 0.16      |
+-----+-----------+-----------+-------------------+-------------------+-----------+
|  4  | 7041.61   | 943.25    | 2.24e-4           | 2.66e-3           | 0.32      |
+-----+-----------+-----------+-------------------+-------------------+-----------+
|  5  | 8807.53   | 1089.76   | 3.39e-5           | 4.01e-4           | 0.06      |
+-----+-----------+-----------+-------------------+-------------------+-----------+

Similarly, the deconvolution RMSE can be printed as a metric of the quality of the fit::

	print(tg.rmse)

Predicting thermograms for other time-temperature histories
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Determining component isotope composition
-----------------------------------------
At this point, the thermogram has been deconvolved into energy complexes 
according to the DAEM and the isotope composition of each energy complex can 
be determined using the `sum_data` file imported previously (see `Importing 
Isotope Data` above). Isotope results are stored in an ``rp.IsotopeResult`` 
class instance.

If the sample was run on the NOSAMS Ramped PyrOx instrument, setting
``blank_corr = True`` and an appropriate value for ``mass_rsd`` will 
automatically blank-correct values according to the blank carbon estimation 
of Hemingway et al. **(in prep)** [11]_. Additionally, setting 
``add_noise = True`` will generate normally distributed uncertainty in 
isotope values using the inputted isotope uncertainty (see `Monte Carlo 
uncertainty estimation` below for further details).

Estimate isotope values using `sum_data`::

	ir = rp.IsotopeResult(sum_data,lt, ec, 
 		blk_corr=True,
 		mass_rsd=0.01,
 		add_noise=False)

You can print the estimates like this::

	ir.summary()

Which prints a table similar to:

+------------------------------------------------------------+
|Isotope and mass estimates for each deconvolved peak:       |
+============================================================+
|NOTE: Combined peak results are repeated in summary table!  |
+-----+--------------------+-------------------+-------------+
|     |      mass (ugC)    |        d13C       |      Fm     |
+-----+--------------------+-------------------+-------------+
|  1  |      84.555698     |     -30.843315    |   0.929585  |
+-----+--------------------+-------------------+-------------+
|  2  |      146.389053    |     -28.449830    |   0.776570  |
+-----+--------------------+-------------------+-------------+
|  3  |      156.773838    |     -25.998722    |   0.460255  |
+-----+--------------------+-------------------+-------------+
|  4  |      127.339722    |     -26.188432    |   0.176751  |
+-----+--------------------+-------------------+-------------+
|  5  |      266.096470    |     -23.059327    |   0.000000  |
+-----+--------------------+-------------------+-------------+
|  6  |      32.907006     |     -24.495371    |   0.058753  |
+-----+--------------------+-------------------+-------------+
|  7  |      33.612607     |     -24.495371    |   0.058753  |
+-----+--------------------+-------------------+-------------+

You can also print the regression RMSEs::
	
	#in python3
	print(ir.RMSEs)


Which results in something similar to:

+------+------------+
|      |    RMSE    |
+======+============+
| mass |  3.536239  |
+------+------------+
| d13C |  0.149527  |
+------+------------+
| Fm   |  0.015916  |
+------+------------+


Kinetic Isotope Effect (KIE)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
While the KIE has no effect on Fm values, as they are fractionation-corrected 
by definition [12]_, the above caclulation explicitly incorporates 
mass-dependent kinetic fractionation effects when calculating stable-carbon 
isotope ratios by using the `DEa` value inputted into the ``rp.EnergyComplex``
instance. While the KIE is potentially important during the pyrolysis of 
organic matter to form hydrocarbons over geologic timescales [10]_, the 
magnitude of this effect is likely minimal within the NOSAMS Ramped PyrOx 
instrument [11]_ and will therefore lead to small corrections in isotope 
values (*i.e.* less than 1 per mille).

Monte Carlo uncertainty estimation
----------------------------------

Saving the output
-----------------




.. Notes and substitutions

.. |realdata| image:: _images/doc_realdata.png

.. |lcurve| image:: _images/doc_Lcurve.png

.. |phis| image:: _images/doc_phis.png

.. |modeleddata| image:: _images/doc_modeleddata.png

.. [1] Note: If analyzing samples run at NOSAMS, all other columns in the `all_data` file generated by LabView are not used and can be deleted or given an arbitrary name.

.. [2] Note: 'd13C_std' and 'Fm_std' default to zero (no uncertainty) if these columns do not exist in the .csv file.

.. [3] Note: For the NOSAMS Ramped PyrOx instrument, plotting against temperature results in a noisy thermogram due to the variability in the ramp rate, dT/dt.

.. [4] Note: Throughout this package, "true" measurements are named with Roman letters -- *e.g.* f (pdf of rates/activation energies), g (fraction of carbon remaining) -- and fitted model variables are named with Greek letters -- *e.g.* phi (sum-of-peak approximation of f), gamma (sum-of-component approximation of g).

.. [5] Braun and Burnham (1999), *Energy & Fuels*, **13(1)**, 1-22 provides a comprehensive review of the kinetic theory, mathematical derivation, and forward-model implementation of the DAEM. 

.. [6] See Forney and Rothman, (2012), *Biogeosciences*, **9**, 3601-3612 for information on building and regularizing a Laplace transform matrix to be used to solve the inverse model using the L-curve method.

.. [7] See White et al., (2011), *J. Anal. Appl. Pyrolysis*, **91**, 1-33 for a review on the KCE and choice of `log10k0`.

.. [8] See Hansen (1994), *Numerical Algorithms*, **6**, 1-35 for a discussion on Tikhonov regularization.

.. [9] Note: Throughout this package, deconvolved rate data peaks are referred to as "peaks", while the forward-modeled components that make-up the thermogram are referred to as "components". This is due to the fact that multiple "peaks" can be combined into a single "component".

.. [10] See Cramer, (2004), *Org. Geochem.*, **35**, 379-392 for a discussion 
	on the relationship between Gaussian Ea peak shape and organic carbon 
	complexity, as well as the KIE.

.. [11] Hemingway et al., (2016), *Radiocarbon*, **in prep** determine that a 
	DEa value of 1.8J/mol best explains the NOSAMS Ramped PyrOx stable-carbon 
	isotope KIE, in addition to determining the blank carbon contribution for 
	this instrument.

.. [12] Stuiver and Polach (1977), *Radiocarbon*, **19(3)**, 355-363 is 
	generally accepted as the standard reference on radiocarbon notation.


