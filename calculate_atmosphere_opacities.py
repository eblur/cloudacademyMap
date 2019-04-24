#! /usr/bin/env python
##
## calculate_atmosphere_opacities.py -- Calculate the opacity,
## dtau/dz, for gas-phase molecules provided by the opacity database
## (from MacDonald et al., in prep). Functions in this file were
## tested in test_gas_opac.ipynb
##
## 2019.01.11 - liac@umich.edu
##
## 2019.04.06 - upgraded to include continuum opacity from CIA and H-
##            - r.macdonald@ast.cam.ac.uk
##-------------------------------------------------------------------

import os
import numpy as np
import scipy.constants as sc
from astropy.io import fits

from maplib import load_out3, cumulative_integral
import opacity_NEW as opac

HOME_DIR = os.environ['HOME'] + '/Dropbox/Science/cloud-academy/Les_Houches_Cloud_Activity/'
DATA_DIR = HOME_DIR + 'static_weather_results/HATP_7b'
OUT_DIR  = HOME_DIR + 'Gas_Ext/'
DB_DIR   = os.environ['HOME'] + '/dev/cloudacademyMap/gas_opac/'

LONS  = np.arange(-180., 180.01, 15) # deg
LATS  = np.arange(0., 67.51, 22.5)   # deg

#---------
# Set up for MacDonald's database

# Grabbed from Readme.txt
RMcD_gas = np.array(['H3+', 'Na', 'K', 'Li', 'Rb', 'Cs', 'H2O', 'CH4', 'NH3', 'HCN', 'CO', \
                     'CO2', 'C2H2', 'H2S', 'N2', 'O2', 'O3', 'OH', 'NO', 'SO2', 'PH3', 'TiO', 'VO', \
                     'AlO', 'SiO', 'CaO', 'TiH', 'CrH', 'FeH', 'ScH', 'AlH', 'SiH', 'BeH', 'CaH', \
                     'MgH', 'LiH', 'SiH', 'CH', 'SH', 'NH', 'Fe', 'Fe+', 'Ti', 'Ti+'])

RMcD_cia = np.array(['H2-H2', 'H2-He'])   # Collisionally-induced absorption pairs

RMcD_gas_upper = np.array([g.upper() for g in RMcD_gas]) 

OPACITY_TREATMENT = 'Log-avg'

#--------
# Supporting functions

def wavel_grid_constant_R(wl_min, wl_max, R):
    """
    Calculate a wavelength grid ranging from wl_min to wl_max at a
    constant spectral resolution R = wl/dwl.
    """

    delta_log_wl = 1.0/R
    N_wl = (np.log(wl_max) - np.log(wl_min)) / delta_log_wl
    N_wl = np.around(N_wl).astype(np.int64)
    log_wl = np.linspace(np.log(wl_min), np.log(wl_max), N_wl)    
    wl = np.exp(log_wl)
    
    return wl

def calc_dtau_dz_gas(gas, opac_dict, ch_dict):
    """
    Calculate dtau/dz for all available gases in the atmosphere.
    """
    n_z   = ch_dict[gas.upper()]  # Number density for this gas (cm^-3)
    sigma = opac_dict[gas]        # Cross section (m^2)
    
    NP  = len(n_z)         # Number of vertical data points
    NWL = len(sigma[0,:])  # Number of wavelengths
    
    result = np.zeros(shape=(NP, NWL))
    
    # For each layer, compute extinction coefficient
    for i in range(NP):
        result[i,:] = n_z[i] * (sigma[i,:] * 1.0e4) # cm^-1

    return result

def calc_dtau_dz_CIA(pair, cia_dict, ch_dict):
    """
    Calculate dtau/dz for CIA pairs.
    
    Only two pairs matter here, so no need for a fancy loop.
    """

    if (pair == 'H2-H2'):
        
        n_1 = ch_dict['H2']  # Number density of first gas in pair (cm^-3)
        n_2 = ch_dict['H2']  # Number density of second gas in pair (cm^-3)
        
    elif (pair == 'H2-He'):
        
        n_1 = ch_dict['H2']  # Number density of first gas in pair (cm^-3)
        n_2 = ch_dict['HE']  # Number density of second gas in pair (cm^-3)
        
    sigma = cia_dict[pair] # Binary cross section (m^5)
    
    NP  = len(n_1)         # Number of vertical data points
    NWL = len(sigma[0,:])  # Number of wavelengths
    
    result = np.zeros(shape=(NP, NWL))
    
    # For each layer, compute extinction coefficient
    for i in range(NP):
        result[i,:] = (n_1[i] * n_2[i]) * (sigma[i,:] * 1.0e10) # cm^-1

    return result

def calc_dtau_dz_H_minus(H_minus_bf, H_minus_ff, n_e, ch_dict):
    """
    Calculate dtau/dz for both sources of H- opacity.
    """
      
    # Load necessary number densities (n_e- through function argument)
    n_H_m = ch_dict['H-']   # Number density of H- (cm^-3)
    n_H = ch_dict['H']      # Number density of H (cm^-3)
    
    sigma_bf = H_minus_bf    # Cross section (m^2)
    sigma_ff = H_minus_ff    # Binary cross section (m^5)
    
    NP  = len(n_H_m)         # Number of vertical data points
    NWL = len(sigma_bf)      # Number of wavelengths
    
    result = np.zeros(shape=(NP, NWL))
    result_bf = np.zeros(shape=(NP, NWL))
    result_ff = np.zeros(shape=(NP, NWL))
    
    # For each layer, compute extinction coefficient
    for i in range(NP):
        result_bf[i,:] = n_H_m[i] * (sigma_bf[:] * 1.0e4) # cm^-1
        result_ff[i,:] = (n_H[i] * n_e[i]) * (sigma_ff[i,:] * 1.0e10) # cm^-1
        
        result[i,:] = result_bf[i,:] + result_ff[i,:]

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
    
    # Convert electron pressure (dyn cm^-2) into electron number density (cm^-3)
    n_e = thermo['pel']/(1.38066e-16 * thermo['T'])   # Hard coded value is Boltzmann constant in cgs
    P = thermo['p']*1.0e-6                            # Convert pressure to bar (expected in opacitiy functions)
    T = thermo['T']
    
    # Array listing CIA pair names
    cia_pairs = RMcD_cia

    # Get the opacities for each gas (loaded into a dictionary)
    cross_sections, CIA, H_minus_bf, H_minus_ff  = opac.Extract_opacity_NEW(np.array(gases), cia_pairs, P, T, 
                                                                            wavel, OPACITY_TREATMENT, root=DB_DIR)
    #opacities = demo.Extract_opacity_PTpairs(np.array(gases), thermo['p'], thermo['T'], wavel,
    #                                         OPACITY_TREATMENT, root=DB_DIR)
    
    dtau_dz = dict()
    
    # First add normal gas cross sections
    for g in gases:
        dtau_dz[g] = calc_dtau_dz_gas(g, cross_sections, Ch_dens)
        
    # Secondly, add CIA opacities
    for pair in cia_pairs:
        dtau_dz[pair] = calc_dtau_dz_CIA(pair, CIA, Ch_dens)
        
    # Finally, add contributions from both sources of H- opacity
    dtau_dz['H-'] = calc_dtau_dz_H_minus(H_minus_bf, H_minus_ff, n_e, Ch_dens)

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
