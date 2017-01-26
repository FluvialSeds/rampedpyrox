from setuptools import setup

def readme():
	with open('README.rst') as f:
		return f.read()

setup(name='rampedpyrox',
	version='0.1.2',
	description='Ramped PyrOx kinetic and isotope analysis',
	long_description=readme(),
	classifiers=[
		'Development Status :: 3 - Alpha',
		'Intended Audience :: Science/Research',
		'License :: Free for non-commercial use',
		'Programming Language :: Python',
		'Programming Language :: Python :: 3',
		'Programming Language :: Python :: 2',
		'Programming Language :: Python :: 3.5',
		'Programming Language :: Python :: 2.7',
		'Topic :: Scientific/Engineering'
	],
	url='https://github.com/FluvialSeds/rampedpyrox',
	download_url='https://github.com/FluvialSeds/rampedpyrox/tarball/0.1.2',
	keywords=[
		'geochemistry',
		'isotopes',
		'kinetics',
		'inverse modeling',
		'carbon cycle'
	],
	author='Jordon D. Hemingway',
	author_email='jhemingway@whoi.edu',
	license='GNU GPL Version 3',
	packages=['rampedpyrox'],
	install_requires=[
		'matplotlib',
		'numpy',
		'pandas',
		'scipy'
	],
	test_suite='nose.collector',
	tests_require=['nose'],
	include_package_data=True,
	zip_safe=False)