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
from matplotlib.colors import LogNorm, SymLogNorm

import astropy.units as u

from maplib import get_wavel, stringy, load_out3

file_root = 'static_weather_results/HATP_7b' # root name for profile folders
nlay  = 53 # number of vertical layers
Teff = 2700.0 # eff/eq temperature

OUTPUT_DIR = 'Grain_Growth/'

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

    p_want : float
       Pressure value desired, must be in the same units as the 'p' column in the data file. 
       Data will be interpolated to this pressure value.

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
            p    = data['p']

            # Add up all the values
            total = np.zeros_like(p)
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

def map_symlognorm(Z, linthresh=0.03, lmin=-3.0, lmax=3.0, cmap=plt.cm.RdBu, **kwargs):
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

    zmap = Z
    levels = np.append(np.append(-np.logspace(np.log10(np.abs(lmin)), np.log10(linthresh), 10)[:-1],
                       np.linspace(-linthresh, linthresh, 3)),
                       np.logspace(np.log10(linthresh), np.log10(lmax), 10)[1:])
    symlog = SymLogNorm(linthresh=linthresh, vmin=lmin, vmax=lmax)
    
    m        = Basemap(projection='kav7', lon_0=0, resolution=None)
    CS_north = m.contourf(X, Y, Z.T, norm=symlog, 
                          levels=levels, extend='both', cmap=cmap, latlon=True)
    CS_south = m.contourf(X, -Y, Z.T, norm=symlog, 
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
    plevels   = [1.e-4, 1.e-3, 1.e-2, 1.0] * u.bar # 0.1, 1, and 10 mbar; then 1 bar (personal interest)
    plabels   = dict(zip(plevels,
                         ['10$^{-4}$ bar','1 mbar','10$^{-2}$ bar','1 bar']))
    
    ## ---- Plot J*
    valnames  = ['J*_C', 'J*_SiO', 'J*_TiO2'] 
    log_Jmin, log_Jmax = -30, 5
    for pw in plevels:
        Jlev  = np.linspace(log_Jmin, log_Jmax, nlev)
        Jstar = interpolate_pressure_map('nuclea', valnames, pw.to('dyne/cm^2').value)
        _J = map_result(Jstar, Jlev, logplot=True, 
                    label=r'$J^{*}$ [cm$^{-3}$ s$^{-1}$]',
                    ticks=np.arange(log_Jmin, log_Jmax+1)[::5])
        plt.title(plabels[pw])
        
        if pw < 1.0*u.bar:
            filename = OUTPUT_DIR + 'jstar_map_{:.1f}mbar.pdf'.format(pw.to(u.bar).value * 1.e3)
        else:
            plt.title('{:.1f} bar'.format(pw.to(u.bar).value))
            filename = OUTPUT_DIR + 'jstar_map_{:.1f}bar.pdf'.format(pw.to(u.bar).value)

        print('Saving {}'.format(filename))
        plt.savefig(filename, format='pdf')
        plt.clf()

    ## ---- Plot number density of cloud particles
    valnames = ['N']
    log_nmin, log_nmax = -3, 6
    for pw in plevels:
        ndlev = np.linspace(log_nmin, log_nmax, nlev)
        ndens = interpolate_pressure_map('dist', valnames, pw.to('bar').value)
        _N = map_result(ndens, ndlev, logplot=True,
                        label=r'$N$ [cm$^{-3}$]',
                        ticks=np.arange(log_nmin, log_nmax+1))
        plt.title(plabels[pw])
                         
        if pw < 1.0*u.bar:
            filename = OUTPUT_DIR + 'ndens_map_{:.1f}mbar.pdf'.format(pw.to(u.bar).value * 1.e3)
        else:
            filename = OUTPUT_DIR + 'ndens_map_{:.1f}bar.pdf'.format(pw.to(u.bar).value)

        print('Saving {}'.format(filename))
        plt.savefig(filename, format='pdf')
        plt.clf()

    ## ---- Plot chinet
    valnames = ['chinet']
    myticks = np.array([-3.0, -1.0, -1.e-3, -1.e-6, -1.e-9, 1.e-9, 1.e-6, 1.e-3, 1.0, 3.0])

    for pw in plevels:
        chimap = interpolate_pressure_map('dust', valnames, pw.to('dyne/cm^2').value)
        _ch = map_symlognorm(chimap, lmin=-1.0, lmax=1.0, linthresh=1.e-9, 
                             label=r'$\chi_{\rm net}$ [cm/s]',
                             ticks=myticks)
        plt.title(plabels[pw])

        if pw < 1.0*u.bar:
            filename = OUTPUT_DIR + 'chinet_map_{:.1f}mbar.pdf'.format(pw.to(u.bar).value * 1.e3)
        else:
            filename = OUTPUT_DIR + 'chinet_map_{:.1f}bar.pdf'.format(pw.to(u.bar).value)

        print('Saving {}'.format(filename))
        plt.savefig(filename, format='pdf')
        plt.clf()

