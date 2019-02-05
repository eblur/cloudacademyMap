#! /usr/bin/env python
##
## map_atm_depth.py -- Map the pressure at which the gas + cloud
## opacity from the atmosphere, tau = 1
##
## 2019.01.22 - liac@umich.edu
##-------------------------------------------------------------------

import numpy as np
import matplotlib.pylab as plt

import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm

from astropy.io import fits
import astropy.units as u
from maplib import load_out3, get_ext_data, cumulative_integral

file_root = 'static_weather_results/HATP_7b' # root name for profile folders
gas_root  = 'Gas_Ext/' # root name for gas opacity (dtau/dz) files
out_root  = 'Cloud_Opt_Depth/' # root name for output files

# Function for reading in the gas opacity file 
#    (results of calculate_atmosphere_opacities.py)
def read_opac_file(filename):
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
    """
    Function for summing each gas contribution to the opacity.

    Parameters
    ----------
    opac_dict : dict
        The dtau_dz values as a function of wavelength and pressure
        layer, for each gas phase element or molecule
    
    keys : list of strings
        The names of the gas or molecules to be incorporated into the
        sum. If None, all available gases in opac_dict will be summed.

    Returns
    -------
    Dictionary of the total dtau_dz as a function of wavelength and
    pressure layer
    """
    if keys is None:
        keys = list(opac_dict.keys())
        
    dtau_dz = np.zeros_like(opac_dict[keys[0]])
    for k in keys:
        dtau_dz += opac_dict[k]
    
    return dtau_dz


def atmosphere_depth(lon, lat, tau=1.0, keys=None):
    """
    Find the atmosphere depth (pressure) at which the cumulative gas
    plus cloud opacity (tau) reaches a chosen value.

    Parameters
    ----------
    lon : float

    lat : float

    tau : float (Default: 1.0)
        Chosen tau value for which the atmosphere depth will be returned

    keys : list of strings
        The names of the gas or molecules to be incorporated into the
        gas opacity calculation. If None, all available gases will be
        summed.

    Returns
    -------
    List of atmospheric pressures (bar) for which tau reaches a threshold
    value, as a function of wavelength.
    """
    # Gas portion
    infile = gas_root + 'Phi{:.1f}Theta{:.1f}_dtau_dz.fits'.format(lon, lat)
    dtau_dz_g = read_opac_file(infile) # dtau/dz for each gas
    dtau_dz_gsum = sum_ext(dtau_dz_g, keys=keys)
    
    # Cloud portion
    zfile = file_root + '_Phi{:.1f}Theta{:.1f}/out3_thermo.dat'.format(lon, lat)
    extfile = file_root + '_Phi{:.1f}Theta{:.1f}/out3_extinction.dat'.format(lon, lat)
    z, dtau_dz_cl = get_ext_data(zfile, extfile)
    
    dtau_dz = dtau_dz_cl + dtau_dz_gsum
    
    thermo = load_out3('thermo', lon, lat)
    z = thermo['z'] # cm
    p_unit = u.dyne / u.cm**2
    p = thermo['p'] * p_unit.to(u.bar)
    wavel = load_out3('wavel', lon, lat)
    
    ci = cumulative_integral(z, dtau_dz)
    result = []
    for i in range(len(wavel)):
        result.append(np.interp(tau, ci[:,i], p))
    return result


##--------------

## Load the data
lons  = np.arange(-180., 180.01, 15) # deg
lats  = np.arange(0., 67.51, 22.5) # deg
nprof = len(lons) * len(lats) # number of (lon, lat) profiles

wavel = load_out3('wavel', lons[0], lats[0], root=file_root)

NLO, NLA, NWA = len(lons), len(lats), len(wavel)
Z = np.zeros((NLO, NLA, NWA))
for i in np.arange(NLO)[:-1]:
    for j in np.arange(NLA):
        d = atmosphere_depth(lons[i], lats[j], tau=1.0) # bar
        Z[i,j,:] = d

# Make +180 slice equal to -180        
Z[-1,:,:] = np.copy(Z[0,:,:])

# Mesh grid to use for the map
X, Y = np.meshgrid(lons,lats)

# Default contour levels to use
nlev = 21
log_pmin, log_pmax = -6, 3
lev  = np.linspace(log_pmin, log_pmax, nlev)

def map_atm_depth(i, levels=lev, cmap=plt.cm.RdYlBu_r):
    """
    Map the gas properties provided in matrix Z for a wavelength value (index i)

    Parameters
    ----------
    i : int
        Index for the wavelength value to be mapped

    levels : numpy.ndarray
        Levels values for the contour map (in logspace)

    cmap : matplotlib colormap    

    Returns
    -------
    Plots the maps and 
    """
    m        = Basemap(projection='kav7', lon_0=0, resolution=None)
    CS_north = m.contourf(X, Y, np.log10(Z[:,:,i].T),
                          levels=levels, extend='both', cmap=cmap, latlon=True)
    CS_south = m.contourf(X, -Y, np.log10(Z[:,:,i].T),
                          levels=levels, extend='both', cmap=cmap, latlon=True)
    
    plt.colorbar(label=r'log $p_{\rm gas}(\tau_{\rm gas}(\lambda) + \tau_{\rm cloud}(\lambda) = 1)$ [bar]',
                 ticks=np.arange(log_pmin+1, log_pmax+1)[::2],
                 orientation='horizontal')
    plt.title('{:.2f} $\mu$m'.format(wavel[i]))
    
    ## -- Plot lat-lon lines on map
    # String formatting function
    def fmtfunc(inlon):
        string = r'{:g}'.format(inlon)+r'$^{\circ}$'
        #print string
        return string

    meridians = np.arange(0., 360., 90)
    m.drawmeridians(meridians, labels=[False,False,False,True],
                    labelstyle=None, fmt=fmtfunc, dashes=[3,3], color='k', fontsize=14)
    parallels = np.arange(-90., 90, 30)
    m.drawparallels(parallels, labels=[True,False,False,False],
                    labelstyle='+/-', dashes=[3,3], color='k',fontsize=14)

    # and we're done!
    return CS_north

## Plot the things!!

for i in range(len(wavel)):
    map_atm_depth(i)
    plt.savefig(out_root + 'atm_depth_{:.2f}um.pdf'.format(wavel[i]))
    plt.clf()

