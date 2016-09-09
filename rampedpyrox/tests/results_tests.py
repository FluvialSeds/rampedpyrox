import numpy as np
import os
import rampedpyrox as rp

#function to load files
def gen_str(name):
	p = os.path.join(os.path.dirname(__file__), name)
	return p

#test bare-bones creation of instance
def test_rpo_Fm_init():
	t_frac = [[1000, 200], [200, 300], [300, 1000]]
	t_frac = np.array(t_frac)
	Fm_frac = [1.0, 0.5, 0.0]

	#create instance
	ri = rp.RpoIsotopes(t_frac = t_frac,
						Fm_frac = Fm_frac)

	assert (ri.Fm_frac == Fm_frac).all()

def test_rpo_t_init():
	t_frac = [[1000, 200], [200, 300], [300, 1000]]
	t_frac = np.array(t_frac)
	Fm_frac = [1.0, 0.5, 0.0]

	#create instance
	ri = rp.RpoIsotopes(t_frac = t_frac,
						Fm_frac = Fm_frac)

	assert (ri.t_frac == t_frac).all()

#test importing form a csv file

def test_rpo_m_from_csv():
	data = gen_str('test_rpo_isotopes.csv')

	#create instance
	ri = rp.RpoIsotopes.from_csv(data,
								blk_corr = True,
								mass_err = 0.01)

	assert hasattr(ri, 'm_frac')
	
def test_rpo_d13C_from_csv():
	data = gen_str('test_rpo_isotopes.csv')

	#create instance
	ri = rp.RpoIsotopes.from_csv(data,
								blk_corr = True,
								mass_err = 0.01)

	assert hasattr(ri, 'd13C_frac')

def test_rpo_Fm_from_csv():
	data = gen_str('test_rpo_isotopes.csv')

	#create instance
	ri = rp.RpoIsotopes.from_csv(data,
								blk_corr = True,
								mass_err = 0.01)

	assert hasattr(ri, 'Fm_frac')

def test_rpo_t_from_csv():
	data = gen_str('test_rpo_isotopes.csv')

	#create instance
	ri = rp.RpoIsotopes.from_csv(data,
								blk_corr = True,
								mass_err = 0.01)

	assert hasattr(ri, 't_frac')


if __name__ == '__main__':

	import nose

	nose.runmodule(
		argv = [__file__, '-vvs', '-x', '--pdb', '--pdb-failure'],
		exit=False)




