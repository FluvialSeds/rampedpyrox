# import matplotlib.pyplot as plt
# import numpy as np
# import pandas as pd

# from numpy.linalg import norm
# from scipy.interpolate import interp1d
# from scipy.optimize import nnls

from rampedpyrox.core.thermogram import (
	RealData,
	ModeledData
	)
from rampedpyrox.core.laplacetransform import (
	calc_A, 
	calc_L_curve, 
	LaplaceTransform
	)