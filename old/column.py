## column.py
## A simple column integrator
## 2018.09.25 - liac@umich.edu
##
## How to use, e.g. to get integrated column of CO. First you need to
## load your data into a dictionary-like object that can be called
## with a key-word string, e.g., data['CO']
##
## from column import integrate_column
## my_integrated_column = integrate_column(data, 'CO')
##
## Requirements:
##    scipy
##------------------------------------------------------

from scipy.integrate import trapz

def integrate_column(data, key):
    """
    Find the radially integrated column for a particular sight line

    Parameters
    ----------
    data : dictionary like object
       data[key] must return an array of numbers

    key : string
       Key-word describe the column you want to retrive from data

    Returns
    -------
    data[key] integrated over data['z'] using trapezoidal
    integration. It is assumed that data[key] has units of cm^-3 and
    data['z'] has units of cm
    """
    z_vals   = data[b'z'] # cm
    key_vals = data[key] # cm^-3
    return trapz(key_vals, z_vals) # cm^2
