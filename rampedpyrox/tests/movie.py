'''
Script for generating activation energy decay movie for PB-60 sample.
'''

import matplotlib as mpl
import matplotlib.animation as an
import matplotlib.pyplot as plt
import numpy as np
import os
import rampedpyrox as rp

#set plotting defaults
mpl.rcParams['mathtext.default']='regular' #math text font
mpl.rcParams['axes.linewidth'] = 0.5 #bounding box line width
mpl.rcParams['xtick.labelsize'] = 14
mpl.rcParams['ytick.labelsize'] = 14
mpl.rcParams['legend.fontsize'] = 8
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['axes.edgecolor'] = 'k'
mpl.rcParams['axes.facecolor'] = 'none'
mpl.rcParams['font.weight'] = 'light'


#function to load files
def gen_str(name):
	p = os.path.join(os.path.dirname(__file__), name)
	return p

data = gen_str('test_data/test_rpo_thermogram.csv')
sum_data = gen_str('test_data/test_rpo_isotopes.csv')
anim_name = gen_str('pb_60_ptE/animation.mp4')

#run through inversion model
tg = rp.RpoThermogram.from_csv(
	data,
	bl_subtract = True,
	nt=250,
	ppm_CO2_err=5,
	T_err=3)

daem = rp.Daem.from_timedata(
	tg,
	log10k0=10,
	Ea_max=350,
	Ea_min=50,
	nEa=400)

ec = rp.EnergyComplex.inverse_model(
	daem, 
	tg,
	combined=None,
	nPeaks='auto',
	omega=0.4477,
	peak_shape='Gaussian',
	thres=0.02)

#calculate the p(t,E) remaining at each timestep:
ptE = ec.f*daem.A

#set-up the figure, axis, etc.
fig, ax = plt.subplots(1, 1, figsize = (7.5, 4.8))

#plot data
line, = ax.plot([], [], 'k', linewidth = '2')

#make text
time_text = ax.text(110, 0.0135, '',fontsize=16)
temp_text = ax.text(110, 0.0122, '',fontsize=16)

#set axis limits and labels
ax.set_xlim([100,300])
ax.set_ylim([0,0.015])

ax.set_xlabel(r'$\mathit{E}$ $(kJ mol^{-1})$')
ax.set_ylabel(r'$\mathit{p}(\mathit{t, E})$')

#remove y tick labels
ax.yaxis.set_ticklabels([])

#set background color for presentation
fig.set_facecolor([253/255,247/255,237/255])
# fig.patch.set_alpha(0.1)

plt.tight_layout()


#initialization function -- plot the background of each frame
def init():
    line.set_data([], [])
    return line,

#animate the function -- this is called sequentially
def animate(i):
    x = ec.Ea
    y = ptE[i,:]

    line.set_data(x, y)
    time_text.set_text('time (sec.) = %.0f' %np.around(tg.t[i],decimals=-1))
    temp_text.set_text(r'Temp ($^{\circ}C$) = %.0f' %np.around(tg.T[i]-273,decimals=-1))

    return line,

# call the animator.  blit=True means only re-draw the parts that have changed.
anim = an.FuncAnimation(
	fig, 
	animate, 
	init_func = init,
	frames = 250, 
	interval = 20, 
	blit = True)

anim.save(
	anim_name, 
	fps = 15, 
	writer = 'mencoder',
	savefig_kwargs={'facecolor':fig.get_facecolor()})
	# extra_args=['-vcodec', 'libx264'])











