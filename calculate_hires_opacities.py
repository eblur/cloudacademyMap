#! /usr/bin/env python
##
## calculate_hires_opacities.py -- Calculate the opacity, dtau/dz, for
##   gas-phase molecules provided by the opacity database (from
##   MacDonald et al., in prep) with a high resolution grid of wavelengths.
##
## Depends on functions in calculate_atmosphere_opacities.py
##
## 2019.02.11 - liac@umich.edu
##------------------------------------------------------------------------

import os
import numpy as np
from astropy.io import fits

from maplib import load_out3, cumulative_integral
import opacity_demo as demo

import calculate_atmosphere_opacities as cao

# Set up paths
HOME_DIR = os.environ['HOME'] + '/Dropbox/Science/cloud-academy/Les_Houches_Cloud_Activity/'
DATA_DIR = HOME_DIR + 'static_weather_results/HATP_7b'
OUT_DIR  = HOME_DIR + 'Gas_Ext/hires_'
DB_DIR   = os.environ['HOME'] + '/dev/cloudacademyMap/gas_opac/'

# Make some pointers for easy reference
RMcD_gas = cao.RMcD_gas
RMcD_gas_upper = cao.RMcD_gas_upper
OPACITY_TREATMENT = cao.OPACITY_TREATMENT

# Longitude, latitude pairs for calculation
LLS   = [(-180.,0.), (-90.,0.), (0.,0.), (90.,0.)] # deg

# Wavelength grid to use
WGRID = np.logspace(np.log10(0.3), 2.0, 1000) # um

def run_hires_calculation(lon, lat, wavel=WGRID):
    """
    Runs the dtau/dz calculation for all available gases in the
    MacDonald databes and saves them to a file. Uses the specified
    wavelength grid instead of the one provided by the
    static_weather_results files.
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

    # Find gases that are also in the MacDonald database
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

    # Get the opacities for each gas (loaded into a dictionary)
    thermo = load_out3('thermo', lon, lat, root=DATA_DIR)
    opacities = demo.Extract_opacity_PTpairs(np.array(gases), thermo['p'], thermo['T'], wavel,
                                             OPACITY_TREATMENT, root=DB_DIR)
    
    dtau_dz = dict()
    for g in gases:
            dtau_dz[g] = cao.calc_dtau_dz(g, opacities, Ch_dens)

    # Save the files for this calculation
    fname = OUT_DIR + 'Phi{:.1f}Theta{:.1f}_dtau_dz.fits'.format(lon, lat)
    cao.write_opac_fits(dtau_dz, fname)
    
    return

##----------------
## Do the things!

if __name__ == '__main__':
    for lon, lat in LLS:
        run_hires_calculation(lon, lat)
