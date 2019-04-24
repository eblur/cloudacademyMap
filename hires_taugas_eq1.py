#! /usr/bin/env python
##
## hires_taugas_eq1.py -- Make high (wavelength) resolution plots of
##    the atmospheric pressure at which tau(gas) = 1.
##
## 2019.02.11 - liac@umich.edu
##-----------------------------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import NullFormatter, MultipleLocator, FormatStrFormatter

# Plotting settings
from matplotlib import rcParams
rcParams['font.family'] = 'sans-serif'
rcParams['xtick.labelsize'] = 10
rcParams['ytick.labelsize'] = 10
rcParams['axes.labelsize'] = 14

from astropy.io import fits
import astropy.units as u
from maplib import load_out3, cumulative_integral, cloud_depth
from calculate_hires_opacities import WGRID, LLS


# Set up paths
HOME_DIR = os.environ['HOME'] + '/Dropbox/Science/cloud-academy/Les_Houches_Cloud_Activity/'
INP_DIR  = HOME_DIR + 'Gas_Ext/hires_'

# Set up colors
'''colors = {'CO':'green', 'CH4':'gray', 'SIO':'xkcd:sky', 
          'H':'red', 'H2S':'xkcd:sun yellow', 'FE':'brown', 
          'TIO':'blue', 'K':'orange', # up to here: from NI in "conventions" doc; from here: using color schemes inspired by Fig 8 (cloud type groupings)
          'VO':'#2c7fb8',  # Metals: Ocean blue to light blue/green
          'ALO':'#41b6c4', 
          'CO2':'#2ca25f', 
          'H2O':'#2c7fb8', # dusty blue
          'NA':'#756bb1',  # grey-purple
          'OH':'#980043',  # deep magenta,
          'MGH':'#fcae91', # peach
          'ALH':'#e34a33', # burnt orange
          'CS':'#cccccc',  # light grey
          'SIH':'#54278f', # deep purple
          'LI':'#b3de69'   # grass green
          }
'''
# Choose a color scheme inspired by Fig 8
colors = {'TIO':'#91003f',  # Metals: Dark magenta to pink
          #'VO':'#ce1256',  
          'VO':'magenta',
          'ALO':'#e7298a', 
          'ALH':'#df65b0', 
          'SIO':'#a63603', # Silicate related stuff (Si, Mg, O): burnt orange to yellow
          'SIH':'#e6550d', 
          'MGH':'#fd8d3c', 
          'H2O':'#0c2c84', # Volatiles: Ocean blue to light blue/green
          'CO2':'#225ea8', 
          'CO':'#1d91c0', 
          'CH4':'#41b6c4', 
          'FEH': 'limegreen',
          'OH':'#7fcdbb',
          'H2S':'#c7e9b4', 
          'CS':'xkcd:sun yellow',
          'H':'#252525',       # Individual atoms: black/grey/purple
          'FE':'crimson', 
          'TI':'royalblue',
          'K':'#54278f', 
          'NA':'#756bb1',
          'LI':'#cbc9e2',
          'H2-H2': 'black',
          'H2-HE': 'dimgrey',
          'H-': 'darkgreen'
          }

labels = {'H2O':'H$_2$O', 'CO2':'CO$_2$', 'CH4':'CH$_4$', 
          'C2H2':'C$_2$H$_2$', 'NH3':'NH$_3$', 'SIO':'SiO', 
          'ALH':'AlH', 'ALO':'AlO', 'MGH':'MgH', 'FEH':'FeH', 
          'H2S':'H$_2$S', 'TIO':'TiO', 'CAH':'CaH', 'NA':'Na',
          'LI':'Li', 'SIH':'SiH', 'H2-H2':'H$_2$-H$_2$',
          'H2-HE':'H$_2$-He', 'H-':'H-', 'FE':'Fe'}

def read_opac_file(filename):
    """
    Read the FITS files output from calculate_atmosphere_opacities.py
    or calculate_hires_opacities.py
    """
    hdulist   = fits.open(filename)
    opac_dict = dict()
    for h in hdulist:
        if h.name == 'PRIMARY':
            pass
        else:
            opac_dict[h.name] = h.data
    hdulist.close()
    return opac_dict

def sum_ext(opac_dict, keys=None):
    """Sum the extinction opacities from a dictionary of dtau/dz

    Parameters
    ----------
    opac_dict : dict
        Contains key-value pairs of chemical species with dtau/dz

    keys : list (Default: None)
        Key values for opac_dict to include in calculation. If None,
        all of the available opacities will be summed.
    """
    if keys is None:
        keys = list(opac_dict.keys())
    dtau_dz = np.zeros_like(opac_dict[keys[0]])
    for k in keys:
        dtau_dz += opac_dict[k]
    return dtau_dz

def atmosphere_depth(lon, lat, tau=1.0, keys=None, wavel=WGRID):
    infile = INP_DIR + 'Phi{:.1f}Theta{:.1f}_dtau_dz.fits'.format(lon, lat)
    dtau_dz_g = read_opac_file(infile) # dtau/dz for each gas
    dtau_dz   = sum_ext(dtau_dz_g, keys=keys)

    thermo = load_out3('thermo', lon, lat)
    z = thermo['z'] # cm
    p_unit = u.dyne / u.cm**2
    p = thermo['p'] * p_unit.to(u.bar)
    
    ci = cumulative_integral(z, dtau_dz)
    result = []
    for i in range(len(wavel)):
        result.append(np.interp(tau, ci[:,i], p))
    return np.array(result)


def plot_depth(ax, ll, keys, wavel, ylim):
    # Plot the gas depth
    for k in keys:
        print(k)
        atm_depth = atmosphere_depth(ll[0], ll[1], keys=[k])
        test = True
        #test = np.min(atm_depth) < ylim[0]
        if test:
            lbl = k
            if k in list(labels.keys()): lbl = labels[k]
            if k in list(colors.keys()):
                ax.plot(wavel, atm_depth, alpha=0.8, color=colors[k], label=lbl)
            else:
                ax.plot(wavel, atm_depth, alpha=0.8, label=lbl)

    # Plot the cloud depth
    p_unit = u.dyne / u.cm**2
    w, cld_depth = cloud_depth('{:.1f}'.format(ll[0]), '{:.1f}'.format(ll[1]), p_val=True)
    ax.plot(w, cld_depth * p_unit.to(u.bar), color='k', ls='--', lw=3, label='Clouds')
    
    xmajorLocator   = MultipleLocator(2.0)
    xmajorFormatter = FormatStrFormatter('%.1f')
    #xminorLocator   = MultipleLocator(0.2)
    xminorFormatter = NullFormatter()

    ax.set_yscale('log')
    ax.set_xscale('log')
    
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    #ax.xaxis.set_minor_locator(xminorLocator)
    ax.xaxis.set_minor_formatter(xminorFormatter)
    
    ax.set_xlabel(r'$\lambda$ [$\mu$m]')
    ax.set_ylabel(r'$p_{\rm gas}(\tau_{x}(\lambda) = 1)$ [bar]')
    ax.set_ylim(ylim)
    ax.set_xlim(0.4, 50.0)
    ax.set_xticks([0.4, 1, 2, 4, 6, 10, 20, 40])
    return

##--------------------
## Do the things!

infile0 = INP_DIR + 'Phi{:.1f}Theta{:.1f}_dtau_dz.fits'.format(LLS[0][0], LLS[0][1])
dtau_dz_g_0 = read_opac_file(infile0) # dtau/dz for each gas



YLIM = [10.0, 1.0e-5]

fig = plt.figure(figsize=(12.8, 9.6))
gs  = GridSpec(2, 2, wspace=0.03)

titles = ['Anti-stellar point',
          r'Morning terminator ($\theta = 0^{\circ}$)',
          'Sub-stellar point', 
          r'Evening terminator ($\theta = 0^{\circ}$)']

#KEYS_TO_PLOT = list(dtau_dz_g_0.keys())
KEYS_TO_PLOT = ['H2O','CO','CO2','CS','TIO','VO','SIO','OH','FEH','ALH','MGH','H2S','NA','K','FE','TI','H2-H2', 'H2-HE','H-']

for i in range(len(titles)):
    print("Running {}".format(titles[i]))
    ax = plt.subplot(gs[i])
    plot_depth(ax, LLS[i], KEYS_TO_PLOT, WGRID, YLIM)
    ax.set_title(titles[i])
    if i in [1,3]:
        ax.set_ylabel('')
        ax.yaxis.set_ticklabels([])
    if i in [0,1]:
        ax.set_xlabel('')
    if i in [2,3]:
        ax.set_ylim(10.0, 3.e-6)
    if i == 2: # the only one with all the elements listed
        ax.legend(loc='upper right', frameon=False, ncol=5)


#plt.tight_layout()
plt.savefig('./gas_opacity_wavel.png', format='png', dpi=500)
#plt.show()
