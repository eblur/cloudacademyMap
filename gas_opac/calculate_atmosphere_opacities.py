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
from maplib import load_out3, cumulative_integral

HOME_DIR = os.environ['HOME'] + '/Dropbox/Science/cloud-academy/Les_Houches_Cloud_Activity/'
DATA_DIR = HOME_DIR + 'static_weather_results/'
OUT_DIR  = HOME_DIR + 'Gas_Ext/'

#---------
# Set up for MacDonald's database

# Grabbed from Readme.txt
RMcD_gas = np.array(['H3+', 'Na', 'K', 'Li', 'Rb', 'Cs', 'H2', 'H2O', 'CH4', 'NH3', 'HCN', 'CO', \
                     'CO2', 'C2H2', 'H2S', 'N2', 'O2', 'O3', 'OH', 'NO', 'SO2', 'PH3', 'TiO', 'VO', \
                     'ALO', 'SiO', 'CaO', 'TiH', 'CrH', 'FeH', 'ScH', 'AlH', 'SiH', 'BeH', 'CaH', \
                     'MgH', 'LiH', 'SiH', 'CH', 'SH', 'NH'])

RMcD_gas_upper = np.array([g.upper() for g in RMcD_gas]) 

#--------
# Supporting functions

