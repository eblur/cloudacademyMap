## cloud_depth.py
## Calculate the cloud depth (where tau = 1) as a function of wavelength.
## 2018.09.26 - liac@umich.edu
##
## How to use:
## First, pick a latitude and longitude value
##
## from cloud_depth import cloud_depth
## w, d_km = cloud_depth(lon, lat) # wavelength (um), depth (km)
##
## Requirements:
##    numpy
##    scipy
##    readinLatLong [provided in Google Drive]
##    read_file [provided in Google Drive]
##----------------------------------------------------------------------------

import numpy as np
from scipy.integrate import trapz
from scipy.interpolate import interp1d

from readinLatLong import get_foldernames
from read_file import read_file

FNAMES = get_foldernames(root='HATP_7b')

def get_data(zfile, extfile):
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

# find point where tau = 1
def calc_cloud_depth(z, integrand, tau=1.0):
    result = []
    for i in range(len(integrand[0])):
        tau_interp = interp1d(integrand[:,i], z)
        result.append(tau_interp(tau))
    return np.array(result) # cm

def cloud_depth(lon, lat):
    """
    Calculates the depth (distance from top of atmosphere) for which the cloud
    opacity becomes optically thick (tau=1)

    Parameters
    ----------
    lon : string
        longitude value (used in filename)

    lat : string
        latitude value (used in filename)

    Returns
    -------
    wavel : numpy.ndarray
        Wavelength of light (um)

    depth : numpy.ndarray
        Distance from top of atmosphere (km), for which tau = 1
    """
    fname   = FNAMES[(lon, lat)]
    zfile   = fname + 'out3_thermo.dat'
    extfile = fname + 'out3_extinction.dat'

    z, dtau_dz = get_data(zfile, extfile)

    wavel = get_wavel(extfile)
    tau_z = cumulative_integral(z, dtau_dz)
    depth_km = calc_cloud_depth(z, tau_z, tau=1.0) * 1.e-5 # km
    return wavel, depth_km
