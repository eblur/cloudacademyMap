#! /usr/bin/env python
##
## calculate_atmosphere_opacities.py -- Calculate the opacity,
## dtau/dz, for gas-phase molecules provided by the opacity database
## (from MacDonald et al., in prep). Functions in this file were
## tested in test_gas_opac.ipynb
##
## 2019.01.11 - liac@umich.edu
##-------------------------------------------------------------------

import os
import numpy as np
from astropy.io import fits

from maplib import load_out3, cumulative_integral
import opacity_demo as demo

HOME_DIR = os.environ['HOME'] + '/Dropbox/Science/cloud-academy/Les_Houches_Cloud_Activity/'
DATA_DIR = HOME_DIR + 'static_weather_results/HATP_7b'
OUT_DIR  = HOME_DIR + 'Gas_Ext/'
DB_DIR   = os.environ['HOME'] + '/dev/cloudacademyMap/gas_opac/'


LONS  = np.arange(-180., 180.01, 15) # deg
LATS  = np.arange(0., 67.51, 22.5) # deg

#---------
# Set up for MacDonald's database

# Grabbed from Readme.txt
RMcD_gas = np.array(['H3+', 'Na', 'K', 'Li', 'Rb', 'Cs', 'H2', 'H2O', 'CH4', 'NH3', 'HCN', 'CO', \
                     'CO2', 'C2H2', 'H2S', 'N2', 'O2', 'O3', 'OH', 'NO', 'SO2', 'PH3', 'TiO', 'VO', \
                     'AlO', 'SiO', 'CaO', 'TiH', 'CrH', 'FeH', 'ScH', 'AlH', 'SiH', 'BeH', 'CaH', \
                     'MgH', 'LiH', 'SiH', 'CH', 'SH', 'NH'])

RMcD_gas_upper = np.array([g.upper() for g in RMcD_gas]) 

OPACITY_TREATMENT = 'Log-avg'

#--------
# Supporting functions

def calc_dtau_dz(gas, opac_dict, ch_dict):
    """
    Calculate dtau/dz for all available gases in the atmosphere.
    """
    n_z   = ch_dict[gas.upper()]
    sigma = opac_dict[gas]
    
    NP  = len(n_z) # number of vertical data points
    NWL = len(sigma[0,:]) # number of wavelengths
    
    # sigma is in units of m^2
    result = np.zeros(shape=(NP, NWL))
    for i in range(NP):
        result[i,:] = n_z[i] * (sigma[i,:] * 1.e4) # cm^-1

    return result

def write_opac_fits(dtau, filename):
    """
    A function for writing dtau_dz (or any other opacity dictionary) to a FITS file.
    """
    hdr = fits.Header()
    hdr['COMMENT'] = "dtau/dz for gas phase elements of HATP-7b"
    hdr['COMMENT'] = "Made from opacity tables of R. MacDonald"
    hdr['COMMENT'] = "See out3_*.dat files for p, T, and z profiles"
    
    primary_hdu = fits.PrimaryHDU(header=hdr)
    hdus_to_write = [primary_hdu]
    for k in dtau.keys():
        hdus_to_write.append(fits.ImageHDU(dtau[k], name=k))
    
    hdu_list = fits.HDUList(hdus=hdus_to_write)
    hdu_list.writeto(filename, overwrite=True)
    return

def run_calculation(lon, lat):
    """
    Runs the dtau_dz calculation for all availabe gases in MacDonald's
    database and saves them to a file.
    """
    # Open the out3_thermo.dat file for a given sight line and grab all of the gases listed
    #thermo = load_out3('thermo', lon, lat, root=DATA_DIR)
    c1 = load_out3('chem1', lon, lat, root=DATA_DIR)
    c2 = load_out3('chem2', lon, lat, root=DATA_DIR)
    c3 = load_out3('chem3', lon, lat, root=DATA_DIR)
    Ch_gas = []
    Ch_dens = {}
    for chem in [c1, c2, c3]:
        for k in chem.keys():
            if k not in ['z','p','T','n<H>']:
                Ch_gas.append(k)
                Ch_dens[k.upper()] = chem[k] # cm^-3

    gases, gases_missing = [], []
    for g in Ch_gas:
        g_up = g.upper()
        if g_up in RMcD_gas_upper:
            ig = np.where(RMcD_gas_upper == g_up)[0]
            gases.append(RMcD_gas[ig][0])
        else:
            gases_missing.append(g)

    print("Gases missing: ", gases_missing)
    print("Gases found: ", gases)

    # Grab the relevant wavelengths
    wavel = load_out3('wavel', lon, lat, root=DATA_DIR)
    thermo = load_out3('thermo', lon, lat, root=DATA_DIR)

    # Get the opacities for each gas (loaded into a dictionary)
    opacities = demo.Extract_opacity_PTpairs(np.array(gases), thermo['p'], thermo['T'], wavel,
                                             OPACITY_TREATMENT, root=DB_DIR)
    
    dtau_dz = dict()
    for g in gases:
            dtau_dz[g] = calc_dtau_dz(g, opacities, Ch_dens)

    # Save the files for this calculation
    fname = OUT_DIR + 'Phi{:.1f}Theta{:.1f}_dtau_dz.fits'.format(lon, lat)
    write_opac_fits(dtau_dz, fname)
    
    return

#----------------

# Run the calculations
if __name__ == '__main__':
    for i in LONS:
        for j in LATS:
            run_calculation(i, j)
