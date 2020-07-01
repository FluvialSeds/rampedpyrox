# Script for Brad to import their RPO data

#import packages
import numpy as np
import pandas as pd
import rampedpyrox as rp

#set time zone relative to GMT
dh = -5 #EST
dy = -66 #to deal with LabView's 66 year future offset

#read the txt file directly and manually tell Pandas to use tab delimiter
df = pd.read_csv('DB-1489-20170302_alt.txt', #file name
	sep = '\t', #tab delimiter. Can also do ' ' or ',' if space- or comma-delimited
	header = None, #if no header exists. If one does exist, then pass header = 0.
	)

#define column header (if it doesn't exist already)
df.columns = [
	'date_time',
	'CO2_scaled',
	'temp',
	'?', #I'm not sure what this column is, but it doesn't matter
	'??', #I'm not sure what this column is, but it doesn't matter
	'???', #I'm not sure what this column is, but it doesn't matter
	]

#drop any NaN rows just to be safe
df = df.dropna(how = 'any')

#now make sure all data are floats
df = df.astype(float)

#make 'date_time' column into timestamp
df['date_time'] = pd.to_datetime(df['date_time'],
	unit = 's'
	)

#now get it into a meaningful time
df['date_time'] = df['date_time'] + pd.Timedelta(weeks = dy*52, hours = dh)

#now reset index to be date_time
df = df.set_index('date_time')

#test it out by making a rampedpyrox thermogram and plotting
tg = rp.RpoThermogram.from_csv(df)
tg.plot()