#! /usr/bin/env python

# G.K.H. Lee : Basic plotting routine to map a variable at a chosen pressure level
# requires the basemap plotting package
# install with anaconda using: conda install -c conda-forge basemap

# 2018.12.20 - Modified by L. R. Corrales (liac@umich.edu)
# To map the cloud growth properties as outlined in Overleaf document (Section)
# 2018.11.09 - Modified to use maplib
# 2018.12.18 - Show pressure instead of cloud depth

# Requires:
# ---------
# numpy, matplotlib, basemap
# Custom library of python functions: 'maplib'

## --------
## TO USE:
##
## In an interactive python session, one can import the main functions:
##
## >>> from map_cloud_depth import map_cloud_depth
## >>> map_cloud_depth[0]
##
## Or you can run the file as a script, and it will save all of the maps to a pdf
## 
## $ ./map_cloud_depth.py
## 
## Be sure to modify the `OUTPUT_DIR` variable to point to your desired output folder
## -------


import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap
from matplotlib.colors import LogNorm

from maplib import get_wavel, stringy, load_out3

file_root = 'static_weather_results/HATP_7b' # root name for profile folders
nlay  = 53 # number of vertical layers
Teff = 2700.0 # eff/eq temperature

OUTPUT_DIR = 'Grain_Growth/'

p_convert = 1.e-6 # bar / (dyne/cm^2)

## ---- Load the data

lons  = np.arange(-180., 180.01, 15) # deg
lats  = np.arange(0., 67.51, 22.5) # deg
nprof = len(lons) * len(lats) # number of (lon, lat) profiles


def interpolate_pressure_map(dtype, values, p_want):
    """
    Calculate the value (or sum of values) at some pressure level in the atmosphere.

    Parameters
    ----------
    dtype : str
       Data file type to use for load_out3 (must contain 'p' values)

    values : list
       List of dictionary keywords to sum. (One value is acceptable, but must be a provided as a list)

    p_want : float (bar)
       Pressure value desired. The data will be interpolated to this value.

    Returns
    -------
    2D map for the (summed) value of interest interpolated to the pressure value at p_want.
    """
    # Read in the cloud depth values for each file.
    # Make a 2D array with dimensions (lon, lat)
    NLO, NLA = len(lons), len(lats)
    result = np.zeros((NLO, NLA))
    for i in range(NLO):
        for j in range(NLA):
            data = load_out3(dtype, lons[i], lats[j])
            p    = data['p'] * p_convert # bar

            # Add up all the values
            total = 0.0
            for v in values:
                total += data[v]

            # interpolate to get the value at the pressure scale we want
            result[i,j] = np.interp(p_want, p, total)
    return result

## ---- Mapping and plotting part

# Mesh grid to use for the map
X, Y = np.meshgrid(lons,lats)

def map_result(Z, levels, cmap=plt.cm.RdYlBu_r, logplot=True, **kwargs):
    """
    Makes a plot of HAT-P-7b at some pressure level
    
    Parameters
    ---------
    levels : numpy.ndarray
        Contour levels to use for the map

    cmap : matplotlib.pyplot.colormap object
        Color map to use

    **kwargs : keyword arguments for plt.colorbar

    Returns
    -------
    Results of Basemap.contourf for the northern hemisphere of the planet

    Also produces a plot.
    """

    if logplot:
        zmap = np.ones_like(Z) * -99.0
        zmap[Z != 0.0] = np.log10(Z[Z != 0.0])
    else:
        zmap = Z
                        
    m        = Basemap(projection='kav7', lon_0=0, resolution=None)
    CS_north = m.contourf(X, Y, zmap.T,
                          levels=levels, extend='both', cmap=cmap, latlon=True)
    CS_south = m.contourf(X, -Y, zmap.T, 
                          levels=levels, extend='both', cmap=cmap, latlon=True)
    
    plt.colorbar(orientation='horizontal', **kwargs)
    
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

## ------ 
## Plot everything and save it, if running this file as a script

if __name__ == '__main__':
    nlev = 21
    plevels   = [1.e-4, 1.e-3, 1.e-2, 1.0] # 0.1, 1, and 10 mbar; then 1 bar (personal interest)

    ## ---- Plot J*
    valnames  = ['J*_C', 'J*_SiO', 'J*_TiO2'] 
    log_Jmin, log_Jmax = -30, 5
    for pw in plevels:
        Jlev  = np.linspace(log_Jmin, log_Jmax, nlev)
        Jstar = interpolate_pressure_map('nuclea', valnames, pw)
        _J = map_result(Jstar, Jlev, logplot=True, 
                    label=r'$J^{*}$ [cm$^{-3}$ s$^{-1}$]',
                    ticks=np.arange(log_Jmin, log_Jmax+1)[::5])

        if pw < 1.0:
            if pw < 1.e-3: plt.title('{:.1f} mbar'.format(pw * 1.e3))
            else: plt.title('{:.0f} mbar'.format(pw * 1.e3))
            filename = OUTPUT_DIR + 'jstar_map_{:.1f}mbar.pdf'.format(pw * 1.e3)
        else:
            plt.title('{:.1f} bar'.format(pw))
            filename = OUTPUT_DIR + 'jstar_map_{:.1f}bar.pdf'.format(pw)

        print('Saving {}'.format(filename))
        plt.savefig(filename, format='pdf')
        plt.clf()

