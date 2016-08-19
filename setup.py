from setuptools import setup

setup(name='rampedpyrox',
      version='0.1',
      description='Ramped PyrOx decolvolution code',
      long_description=readme(),
      classifiers=[
            'Development Status :: 1 - Planning',
            'Intended Audience :: Science/Research',
            'License :: Free for non-commercial use',
            'Programming Language :: Python :: 3.5',
            'Topic :: Scientific/Engineering'
      ],
      url='http://github.com/put_url_here',
      author='Jordon D. Hemingway',
      author_email='jhemingway@whoi.edu',
      license='MIT',
      packages=['rampedpyrox'],
      install_requires=[
      	'numpy',
      	'pandas',
      	'scipy'
      ],
      test_suite='nose.collector',
      tests_require=['nose'],
      include_package_data=True,
      zip_safe=False)