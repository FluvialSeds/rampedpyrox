# timedata module tests

import numpy as np
import os
import rampedpyrox as rp

#function to load files
def gen_str(name):
	p = os.path.join(os.path.dirname(__file__), name)
	return p
