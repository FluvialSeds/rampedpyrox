'''
This package contains all the exceptions for the rampedpyrox package.
'''

#define a core exception class for subclassing
class rpException(Exception):
	'''
	Root Exception class for the rampedpyrox package. Do not call directly.
	'''
	pass


class ArrayError(rpException):
	'''
	Array-like object is not in the right form (e.g. strings).
	'''
	pass


class FileError(rpException):
	'''
	If a file does not contain the correct data.
	'''
	pass


class LengthError(rpException):
	'''
	Length of array is not what it should be.
	'''
	pass


class ScalarError(rpException):
	'''
	If something is not a scalar.
	'''
	pass


class StringError(rpException):
	'''
	If a string is not right.
	'''
	pass