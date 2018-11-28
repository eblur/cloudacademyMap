## read_file.py
## Reads in static_weather output files
## Created for Les Houches Cloud Academy
## 2018.09.25 - munazza.alam@cfa.harvard.edu
## 2018.09.26 - liac@umich.edu fixed a bug
##
## Requirements:
##    numpy
##------------------------------------------------------

import numpy as np

def read_file(fname):
    '''
    Read in static_weather output files & write data values to a dictionary

    Parameters
    ----------
    fname : string
        full path to output data file from static_weather

    Returns
    -------
    data_dict : dictionary like object
        keys are column names from input file & associated values are
        data columns from input file
    '''

    #read in static_weather output file
    names = np.genfromtxt(fname,delimiter='',skip_header=3,comments='#',dtype=None)
    data = np.genfromtxt(fname,delimiter='',skip_header=4,comments='#',dtype=None)

    keys = names[0] #column names
    values = [data[:,i] for i in range(len(keys))]
    data_dict = dict(zip(keys,values))

    return data_dict
