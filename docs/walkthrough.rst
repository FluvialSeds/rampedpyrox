Comprehensive Walkthrough
=========================
The following examples should form a comprehensive walkthough of downloading
the package, getting thermogram data into the right form for importing,
running the DAEM inverse model, peak-fitting the f(Ea) distribution,
determining the isotope composition of each Ea Gaussian peak, and performing
Monte Carlo uncertainty estimates.

Downloading the package
-----------------------

Using the ``pip`` package manager
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
(coming with first full release)
``rampedpyrox`` and the associated dependent packages can be downloaded
directly from the command line using ``pip``::

	pip install rampedpyrox


Downloading from source
~~~~~~~~~~~~~~~~~~~~~~~
Alternatively, ``rampedpyrox`` source code can be downloaded directly from
`my github repo <http://github.com/FluvialSeds/rampedpyrox>`_. Or, if you have
git installed::

	git clone git://github.com/FluvialSeds/rampedpyrox.git

And keep up-to-date with the latest version by doing::

	git pull

from within the rampedpyrox directory.

Dependencies
~~~~~~~~~~~~
The following packages are required to run ``rampedpyrox``:

* `python <http://www.python.org>`_ >= 2.7, including Python 3.x

* `matplotlib <http://matplotlib.org>`_ >= 1.5.2

* `numpy <http://www.numpy.org>`_ >= 1.11.1

* `pandas <http://pandas.pydata.org>`_ >= 0.18.1

* `scipy <http://www.scipy.org>`_ >= 0.18.0


Getting data in the right format
--------------------------------

Importing thermogram data
~~~~~~~~~~~~~~~~~~~~~~~~~
For thermogram data (*i.e.* `all_data` .csv files), this package requires that
the first column be a `datetime` index in an *hh : mm : ss AM/PM* format and 
that the file contains 'CO2_scaled' and 'temp' columns. For example:

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

All other columns generated in an `all_data` file on the NOSAMS Ramped PyrOx
instrument are not used and can be deleted or renamed.

Once the file is in this format, generate a string pointing to it like this::

	#create string of path to data
	all_data = '/path_to_folder_containing_data/all_data.csv'

Importing isotope data
~~~~~~~~~~~~~~~~~~~~~~
For isotope data (*i.e.* `sum_data` .csv files) this package requires that the
first two rows correspond to the starting time of the experiment and the
initial trapping time of fraction 1, respectively. Additionally, `sum_data`
must contain the columns 'fraction', 'ug_frac', 'd13C', 'd13C_std', 'Fm',
and 'Fm_std' [1]_. For example:

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

Note that the `date_time` value for fraction '-1' is the same as the 
`date_time` value for the first row in `all_data`, and that the value for
fraction '0' is the initial time when trapping for fraction 1 began.

Once the file is in this format, generate a string point to it like this::

	#create string of path to data
	sum_data = '/path_to_folder_containing_data/sum_data.csv'

Making a RealData instance
--------------------------
Once these strings have been defined, you are ready to import the package
and generate a ``rp.RealData`` instance containing the thermogram data::

	#load modules
	import rampedpyrox as rp

	#number of timepoints to be used in down-sampled thermogram
	nT = 250

	#save to RealData instance
	rd = rp.RealData(all_data, nT=nT)

Plot the thermogram against temperature [2]_ or time::

	#load modules
	import matplotlib.pyplot as plt

	#make a figure
	fig,ax = plt.subplots(1,2)

	#plot results
	ax[0] = rd.plot(ax=ax[0], xaxis='time')
	ax[1] = rd.plot(ax=ax[1], xaxis='temp')

Resulting plot looks like this:

|realdata|

Generating the f(Ea) distribution
---------------------------------

Deconvolving f(Ea) into Gaussians
---------------------------------


Determining peak isotope composition
------------------------------------

Monte Carlo uncertainty estimation
----------------------------------

Saving the output
-----------------


.. |realdata| image:: _images/doc_realdata.png

.. [1] Note: 'd13C_std' and 'Fm_std' are unused if passed into an 
	``rp.IsotopeResult`` instance with ``add_noise=False``.

.. [2] Note: For the NOSAMS Ramped PyrOx instrument, plotting against temp.
	results in a noisy thermogram due to the variability in the ramp rate,
	dT/dt.
