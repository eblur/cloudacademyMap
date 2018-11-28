# G.K.H. Lee : Basic plotting routine to map a variable at a chosen pressure level
# requires the basemap plotting package
# install with anaconda using: conda install -c conda-forge basemap

import numpy as np
import matplotlib.pylab as plt
import matplotlib.cm as cm
from mpl_toolkits.basemap import Basemap

# Basic input info -
# file name, number of vertical layers, number of profiles, eff/eq temperature
fname = 'PTprofiles-HAT-P-7b-Solar-NoDrag-nit-1036800-Av-Helling.dat'
nlay = 53
nprof = 16
Teff = 2700.0

# Choose pressure (bar) to plot
P_plot = 1.0e-1

# Load data into master array
data = np.loadtxt(fname,delimiter=',',skiprows=8,comments=['#',' '])

# Deconstruct data into familar variables and dimensionalise arrays
lon = np.zeros(nprof)         #longitude (degrees)
lat = np.zeros(nprof)         #latitude (degrees)
P = np.zeros((nprof,nlay))    #gas pressure (bar)
T = np.zeros((nprof,nlay))    #gas temperature (K)
U = np.zeros((nprof,nlay))    #horizontal velocity (m s-1)
V = np.zeros((nprof,nlay))    #meridonial velocicty (m s-1)
W_Pa = np.zeros((nprof,nlay)) #pressure based vertical velocity (Pa s-1)
W = np.zeros((nprof,nlay))    #vertical velocity (m s-1)

# Loop over all profiles and add data to arrays
didx = 0 # 1D data index
for p in range(nprof):
    for l in range(nlay):
        # Add data to arrays
        lon[p] = data[didx,1]
        lat[p] = data[didx,2]
        P[p,l] = data[didx,3]
        T[p,l] = data[didx,4]
        U[p,l] = data[didx,5]
        V[p,l] = data[didx,6]
        W_Pa[p,l] = data[didx,7]
        W[p,l] = data[didx,8]
        # Add 1 to data index
        didx = didx + 1

# Remove duplicate lat and lon
lon = np.unique(lon)
lat = np.unique(lat)

# add an additional longitude point at 180 degrees = same profile as -180
lons = np.zeros(len(lon)+1)
lons[:-1] = lon
lons[-1] = 180.0
lats = lat

## Mapping and plotting part ##

# Z is our mapping variable - choose what variable is to be interpolated and mapped
Z_a = np.zeros((nprof,nlay))
Z_a = T
Z = np.zeros((len(lats),len(lons)))

# We have a pressure vertical coordinate which makes things easier
# we can directly interpolate Z_a to P_plot
p = 0
for l1 in range(len(lons)):
    # Repeat first profile for last profile points
    if l1 == len(lons)-1:
      p = 0
    for l2 in range(len(lats)):
        Z[l2,l1] = np.interp(np.log10(P_plot),np.log10(P[p,:]),Z_a[p,:])
        p = p + 1

# X,Y grid is length lons * lats - Z array has to be shape (len(Y),len(X))
X, Y = np.meshgrid(lons,lats)

# Initilise figure object
fig = plt.figure()
ax = fig.add_subplot(111)

# we use the basemap package for easier global variable plotting
# install with anaconda using: conda install -c conda-forge basemap
# for availible projections see: https://matplotlib.org/basemap/users/mapsetup.html
# lon_0 is the longitude of the zero (center) point
m = Basemap(projection='kav7',lon_0=0,resolution=None)

# Find levels between max and min of mapped variable
nlev = 20
levels = np.linspace(np.amin(Z),np.amax(Z),nlev)

# Colour map
cmap = plt.cm.RdYlBu_r

# Plot data as filled contour
CS = m.contourf(X,Y,Z,levels=levels,extend='both',cmap=cmap,latlon=True)

# Colour bar and formatting
for c in CS.collections:
    c.set_edgecolor("face")
CB = plt.colorbar(CS, orientation='horizontal', format="%d")
CB.set_label(r'T$_{\rm gas}$ [K]', size=16)
CB.ax.tick_params(labelsize=12)

# Title is chosen pressure level
plt.title('{:0.1f}'.format(P_plot*1.0e3) + ' mbar',fontsize=16)

# Increase size of labels
plt.tick_params(axis='both', which='major', labelsize=16)

# String formatting function
def fmtfunc(inlon):
    string = r'{:g}'.format(inlon)+r'$^{\circ}$'
    #print string
    return string

# Plot lat-lon lines on map
color1='k'
meridians = np.arange(0.,360.,90)
m.drawmeridians(meridians,labels=[False,False,False,True],labelstyle=None,fmt=fmtfunc,dashes=[3,3],color=color1,fontsize=16)
parallels = np.arange(-90.,90,30)
m.drawparallels(parallels,labels=[True,False,False,False],labelstyle='+/-',dashes=[3,3],color=color1,fontsize=16)

# Save figure
plt.tight_layout(pad=1.05, h_pad=None, w_pad=None, rect=None)
plt.savefig('HAT-P-7b_T_map_initial.pdf',dpi=144,bbox_inches='tight')

# Show figure
plt.show()
