##
## maplib.py
## Consolidated set of functions useful for mapping the atmosphere of
## HATP-7b, information contained in static_weather_results folder
##
## Created 2018.11.08 - liac@umich.edu
##
## Requirements:
##    numpy, scipy, glob, re
##
## All packages can be installed using conda
##---------------------------------------------------------------------

import numpy as np
import glob
import re

from scipy.integrate import trapz
from scipy.interpolate import interp1d

# Change this string to match your directory structure
FOLDER_ROOT = 'static_weather_results/HATP_7b'

# Functions that will be imported when you run:
# >>> from maplib import *
__all__ = ['get_foldernames', 'read_file', 'integrate_column', \
           'get_ext_data', 'get_wavel', 'cumulative_integral', \
           'cloud_depth', 'load_out3', 'get_rhod']

## ----- Contents of readinLatLong.py (c.j.baxter@uva.nl)

# Function which cuts N characters of a string after a substrong
def cut_string(fullstring, substring, Nchar):
    newstring = fullstring[fullstring.index(substring):
                        fullstring.index(substring) + Nchar]
    return newstring

# Function to find the numbers in a string
def find_numbers(string, ints=True):
    numexp = re.compile(r'[-]?\d[\d,]*[\.]?[\d{2}]*') #optional - in front
    numbers = numexp.findall(string)
    numbers = [x.replace(',','') for x in numbers]
    if ints:
        res =  [int(x.replace(',','').split('.')[0]) for x in numbers][0]
    else:
        res =  numbers[0]
    return res

# Retrieves folder names for every available latitude an longitude
# (aka all the things)
def get_foldernames(root=FOLDER_ROOT):
    """
    Reads in all the available folder names in order to find latitutde,
    longitude, and corresponding folder name.

    Parameters
    ----------
    root : String
        Root folder and filename prefix

    Returns
    -------
    A dictionary where the keys are tuples of strings (latitude, longitude),
    and the values are the strings with the associated folder name.
    """
    # All the folders containing drift files
    folders = glob.glob(root + '*')
    #print(folders)

    # Loop over all the folders and creatte a list of lattitudes and longitudes
    latlong = []
    for folder in folders:
        # Hacky job with string split method
        f = folder.split('Phi')
        coords = f[1].split('Theta')
        phi, theta = coords
        #phi = find_numbers(cut_string(folder,'Phi', 7), ints=False)
        #theta = find_numbers(cut_string(folder,'Theta', 9), ints=False)
        latlong.append((phi, theta))

    fnames = []
    for phi, theta in latlong:
        fnames.append(root + '_Phi{}Theta{}/'.format(phi,theta))

    return dict(zip(latlong, fnames))


# ---- Contents of read_file.py (munazza.alam@cfa.harvard.edu)

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
    names = np.genfromtxt(fname,delimiter='',skip_header=3,comments='#',dtype=None,encoding=None)
    data = np.genfromtxt(fname,delimiter='',skip_header=4,comments='#',dtype=None,encoding=None)
    
    keys = names[0] #column names
    values = [data[:,i] for i in range(len(keys))]
    data_dict = dict(zip(keys,values))
            
    return data_dict 

# ---- Contents of column.py (liac@umich.edu)

# A simple column integrator
def integrate_column(data, key, is_dens=True):
    """
    Find the radially integrated column for a particular sight line

    Parameters
    ----------
    data : dictionary like object
       data[key] must return an array of numbers.
       This should work on astropy.Table objects

    key : string
       Key-word describe the column you want to retrive from data

    is_dens : bool 
       True (default) if the values represent a density. Output will be and
       integrated column density.
       False if the values do not represent a density.

    Returns
    -------
    If is_dens = True: Returns integral of data[key] over data['z']

    if is_dens = False: Returns the integral of data[key]/dp over data['p'], 
    where dp = data['p'][1:] - data['p'][:-1]
    """
    key_vals = data[key]
    if is_dens:
        z_vals = data['z'] # cm
        return trapz(key_vals, z_vals)
    else:
        p_vals = data['p'] # dyne/cm^2
        dp  = p_vals[1:] - p_vals[:-1]

        # Construct the function to integrate
        to_int = np.copy(key_vals[:-1])
        to_int[dp == 0.0] = 0.0
        to_int[dp != 0.0] /= dp[dp != 0.0]

        # This might be slightly redundant. Need to test.
        return trapz(to_int, p_vals[:-1])


## ---- Contents of cloud_depth.py (liac@umich.edu)

FNAMES = get_foldernames(root=FOLDER_ROOT)

def get_ext_data(zfile, extfile):
    # get some atmosphere column parameters
    # removed reliance on astropy
    # tbl = Table.read(zfile, format='ascii', header_start=2, data_start=3)
    tbl = read_file(zfile)
    z   = np.array(tbl['z']) # cm
    rho = np.array(tbl['rho']) # g cm^-3
    
    # load data from extinction file
    extdat = np.loadtxt(extfile, skiprows=4)
    # cm^2 g^-1 (cross section per unit atmospheric mass)
    # typical dust-to-gas ratio is 1.e-4
    kappa = extdat[:,1:]

    # dtau/dz = rho x kappa
    rho2d = np.tile(rho, (len(kappa[0,:]), 1)).T
    dtau_dz = kappa * rho2d

    return z, dtau_dz

# get the wavelengths
def get_wavel(filename):
    f = open(filename, 'r')
    _ = f.readline()
    _ = f.readline()
    wavel_text = f.readline().split()
    return np.array([np.float(w) for w in wavel_text])

# calculate a cumulative integral
def cumulative_integral(z, ext):
    z2d = np.tile(z, (len(ext[0,:]), 1)).T
    result = np.zeros(shape=(len(z), len(ext[0])))
    for i in np.arange(1, len(z), dtype=int):
        result[i,:] = trapz(ext[:i+1,:], z2d[:i+1,:], axis=0)
    return result

def cloud_depth(lon, lat, tau=1.0):
    """
    Calculates the depth (distance from top of atmosphere) for which
    the cloud opacity becomes optically thick (tau=1) or hits some
    other threshold tau value

    Parameters
    ----------
    lon : string
        longitude value (used in filename)

    lat : string
        latitude value (used in filename)

    tau : float  (Default: 1.0)
        Target tau value

    Returns
    -------
    wavel : numpy.ndarray
        Wavelength of light (um)

    depth : numpy.ndarray of size len(wavel)
        Distance from top of atmosphere (km), for which tau = 1
    """
    fname   = FNAMES[(lon, lat)]
    zfile   = fname + 'out3_thermo.dat'
    extfile = fname + 'out3_extinction.dat'

    z, dtau_dz = get_ext_data(zfile, extfile)

    wavel = get_wavel(extfile)
    tau_z = cumulative_integral(z, dtau_dz)

    #depth_km = calc_cloud_depth(z, tau_z, tau=1.0) * 1.e-5 # km
    
    # interpolate over the cumulative integral to get the depth at
    # which tau reaches the desired threshold
    result = []
    for i in range(len(tau_z[0])):
        tau_interp = interp1d(tau_z[:,i], z, 
                              fill_value=(z[0], z[-1]), bounds_error=False) #^(1)
        result.append(tau_interp(tau))
    # (1) In some cases, the integrated optical depth never reaches the target value
    # Return z[-1] (the deepest part of the atmosphere) in that case
    return wavel, np.array(result) * 1.e-5 # micron, km

## ---- Convenience function for formatting latitudes and longitudes

# silly function for handling lat and long strings
def stringy(val):
    return '{:.1f}'.format(val)

# ---- Convenience functions for loading specific files

OUT3_FTYPES = ['albedo', 'chem1', 'chem2', 'chem3', 'dist', 'dust', \
               'thermo', 'imp', 'nuclea', 'extinction', 'wavel']

# convenience function for quickly loading output files
def load_out3(type, lon, lat, root=FOLDER_ROOT):
    """
    Load one of the out3 files from static_weather_output

    Parameters
    ----------
    type : string
        Type of output file to load (see OUT3_FTYPES)

    lon : float
        Longitude value to retrieve

    lat : float
        Latitude value to retrieve

    root : string
        Root string for specifying location of the static_weather_output files

    Returns
    -------
    Dictionary-like object where the keys are the column names for each file.
    Exceptions:
        type == 'extinction' will return (z, dtau_dz). See maplib.get_ext_data
        type == 'wavel' will return wavelengths from out3_thermo. See maplib.get_wavel
    """
    assert type in OUT3_FTYPES

    froot = root + '_Phi{:.1f}Theta{:.1f}/out3_{}.dat'
    
    # extinction files have a different format from the rest
    if type == 'extinction':
        zfile   = froot.format(lon, lat, 'thermo')
        extfile = froot.format(lon, lat, 'extinction')
        return get_ext_data(zfile, extfile)
    elif type == 'wavel':
        extfile = froot.format(lon, lat, 'extinction')
        return get_wavel(extfile)
    else:
        fname = froot.format(lon, lat, type)
        return read_file(fname)

## ---- Added 2018.11.24

# Returns the adjusted rho_d column, so that
# rho_d = 0 when there are no dust grains (a = 0)
def get_rhod(lon, lat):
    dust = load_out3('dust', lon, lat)
    thermo = load_out3('thermo', lon, lat)
    return dust['rhod/rho'] * thermo['rho']

