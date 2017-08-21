'''
rampedpyrox was created as a thesis dissertation supplement by:

	Jordon D. Hemingway, MIT/WHOI Joint Program
	Currently Postdoctoral Fellow, Harvard University
	jordon_hemingway@fas.harvard.edu

source code can be found at:
	
	https://github.com/FluvialSeds/rampedpyrox

documentation can be found at:

	http://rampedpyrox.readthedocs.io

Version 1.0.1 is current as of 21 August 2017 and reflects the notation used
in Hemingway et al. Biogeosciences, 2017.

To do for future versions:
* Change exception structure to "try/except"
* Add more robust nose testing suite for debugging

'''

from __future__ import(
	division,
	print_function,
	)

__version__ = '1.0.1'

__docformat__ = 'restructuredtext en'


#import timedata classes
from .timedata import(
	RpoThermogram,
	)

#import model classes
from .model import(
	Daem,
	)

#import ratedata classes
from .ratedata import(
	EnergyComplex,
	)

#import results classes
from .results import(
	RpoIsotopes,
	)

#import package-level functions
from .core_functions import(
	assert_len,
	calc_L_curve,
	derivatize,
	extract_moments,
	plot_tg_isotopes,
	)
