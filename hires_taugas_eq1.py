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
colors = {'CO':'green', 'CH4':'gray', 'SIO':'xkcd:sky', 
          'H':'red', 'H2S':'xkcd:sun yellow', 'FE':'brown', 
          'TIO':'blue', 'K':'orange', # up to here: from NI in "conventions" doc
          'CO2':'#2ca25f', # green hue
          'H2O':'#2c7fb8', # dusty blue
          'NA':'#756bb1',  # grey-purple
          'OH':'#980043',  # deep magenta,
          'MGH':'#fcae91', # peach
          'ALH':'#e34a33', # burnt orange
          'CS':'#cccccc',  # light grey
          'VO':'#dd1c77',  # bright pink
          'SIH':'#54278f', # deep purple
          'ALO':'#ffffb3', # banana
          'LI':'#b3de69'   # grass green
          }

labels = {'H2O':'H$_2$O', 'CO2':'CO$_2$', 'SIO':'SiO', 
          'ALH':'AlH', 'ALO':'AlO', 'MGH':'MgH', 'FEH':'FeH', 
          'H2S':'H$_2$S', 'TIO':'TiO', 'CAH':'CaH', 'NA':'Na', 
          'LI':'Li', 'SIH':'SiH'}

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

    Paremeters
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
        if np.min(atm_depth) < ylim[0]:
            lbl = k
            if k in list(labels.keys()): lbl = labels[k]
            if k in list(colors.keys()):
                ax.plot(wavel, atm_depth, alpha=0.8, color=colors[k], label=lbl)
            else:
                ax.plot(wavel, atm_depth, alpha=0.8, label=lbl)

    # Plot the cloud depth
    p_unit = u.dyne / u.cm**2
    w, cld_depth = cloud_depth('{:.1f}'.format(ll[0]), '{:.1f}'.format(ll[1]), p_val=True)
    ax.plot(w, cld_depth * p_unit.to(u.bar), alpha=0.5, color='k', lw=3, label='Clouds')

    ax.set_yscale('log')
    ax.set_xscale('log')
    ax.set_xlabel(r'$\lambda$ [$\mu$m]')
    ax.set_ylabel(r'$p_{\rm gas}(\tau_{v,x}(\lambda) = 1)$ [bar]')
    ax.set_ylim(ylim)
    ax.set_xlim(0.3, 49.0)
    ax.legend(loc='upper right', frameon=False, ncol=3)
    return

##--------------------
## Do the things!

infile0 = INP_DIR + 'Phi{:.1f}Theta{:.1f}_dtau_dz.fits'.format(LLS[0][0], LLS[0][1])
dtau_dz_g_0 = read_opac_file(infile0) # dtau/dz for each gas


YLIM = [0.1, 3.e-6]

fig = plt.figure(figsize=(12.8, 9.6))
gs  = GridSpec(2, 2, wspace=0.03)

titles = ['Anti-stellar point',
          r'Morning terminator ($\theta = 0^{\circ}$)',
          'Sub-stellar point', 
          r'Evening terminator ($\theta = 0^{\circ}$)']

for i in range(len(titles)):
    print("Running {}".format(titles[i]))
    ax = plt.subplot(gs[i])
    plot_depth(ax, LLS[i], list(dtau_dz_g_0.keys()), WGRID, YLIM)
    ax.set_title(titles[i])
    if i in [1,3]:
        ax.set_ylabel('')
        ax.yaxis.set_ticklabels([])
    if i in [0,1]:
        ax.set_xlabel('')

plt.tight_layout()
plt.savefig("gas_opacity_wavel.png", format='png')
#plt.show()
