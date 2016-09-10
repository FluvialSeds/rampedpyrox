'''
Rampedpyrox module

.. moduleauthor:: Jordon D. Hemingway <jordonhemingway@gmail.com>
'''

#import exceptions
from .core.exceptions import(
	rpException,
	ArrayError,
	FileError,
	FitError,
	LengthError,
	RunModelError,
	ScalarError,
	)

#import timedata classes
from .timedata.timedata import(
	RpoThermogram,
	)

#import model classes
from .model.model import(
	Daem,
	)

#import ratedata classes
from .ratedata.ratedata import(
	EnergyComplex,
	)

#import results classes
from .results.results import(
	RpoIsotopes,
	)

#import package-level functions
from .core.core_functions import(
	assert_len,
	calc_L_curve,
	derivatize,
	)
