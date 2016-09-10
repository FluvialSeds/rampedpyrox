'''
.. moduleauthor:: Jordon D. Hemingway <jordonhemingway@gmail.com>
'''

from __future__ import(
	division,
	print_function,
	)

__version__ = '0.0.2'

__docformat__ = 'restructuredtext en'

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
