# Demostrates how to open the opacity database and interpolate to a given P, T, and wavelength grid
# Author: Ryan J. MacDonald - 8th November, 2018

import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d as gauss_conv
from numba.decorators import jit
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter

#***** Begin function declarations *****# 

@jit(nopython = True)
def prior_index(vec, value, start):
    
    '''Finds the index of a grid closest to a specified value'''
    
    value_tmp = value
    
    if (value_tmp > vec[-1]):
        return (len(vec) - 1)
    
    # Check if value out of bounds, if so set to edge value
    if (value_tmp < vec[0]): value_tmp = vec[0]
    if (value_tmp > vec[-2]): value_tmp = vec[-2]
    
    index = start
    
    for i in range(len(vec)-start):
        if (vec[i+start] > value_tmp): 
            index = (i+start) - 1
            break
            
    return index

@jit(nopython=True)
def prior_index_V2(val, grid_start, grid_end, N_grid):
    
    if (val < grid_start): 
        return 0
    
    elif (val > grid_end):
        return N_grid-1
    
    else:
        i = (N_grid-1) * ((val - grid_start) / (grid_end - grid_start))
        return int(i)

@jit(nopython=True)
def closest_index(val, grid_start, grid_end, N_grid):
    
    '''Finds the index of a UNIFORM grid closest to a specified value.
    
       Assuming a uniform grid dramatically speeds calculation'''
    
    if (val < grid_start): 
        return 0
    
    elif (val > grid_end):
        return N_grid-1
    
    else:
        i = (N_grid-1) * ((val - grid_start) / (grid_end - grid_start))
        if ((i%1)<=0.5):
            return int(i)
        else:
            return int(i)+1
                        
@jit(nopython = True)
def P_interpolate_wl_initialise(N_T, N_P, N_wl, log_sigma,
                                nu_l, nu_model, nu_r, nu_opac, N_nu,
                                x, b1, b2, mode):
    
    '''Interpolates raw cross sections onto the model P and wl grid.
       
       Note: input sigma has format log10(cross_sec)[log(P)_grid, T_grid, nu_grid], 
             whilst output has format cross_sec[T, wl_model]
             
       The input is in wavenumnber to take advantage of fast prior index location 
       on a uniform grid, which wouldn't work for the (non-uniform) wavelength grid
       Array reversal to output in increasing wavelength is handled by indexing
       by a factor of (N_wl-1)-k throughout 
    '''
    
    sigma_pre_inp = np.zeros(shape=(N_T, N_wl))
    
    N_nu_opac = len(nu_opac)   # Number of wavenumber points in sigma array
    
    for k in range(N_nu): # Note that the k here is looping over wavenumber
        
        # Indicies in pre-computed wavenumber array of LHS, centre, and RHS of desired wavenumber grid
        z_l = closest_index(nu_l[k], nu_opac[0], nu_opac[-1], N_nu_opac)
        z = closest_index(nu_model[k], nu_opac[0], nu_opac[-1], N_nu_opac)
        z_r = closest_index(nu_r[k], nu_opac[0], nu_opac[-1], N_nu_opac)
        
        for j in range(N_T):
            
            # If nu (wl) point out of range of opacity grid, set opacity to zero
            if ((z == 0) or (z == (N_nu_opac-1))):
                sigma_pre_inp[j, ((N_wl-1)-k)] = 0.0
                
            else:
                            
                # Opacity sampling
                if (mode == 1):
                    
                    # If pressure below minimum, set to value at min pressure
                    if (x == -1):
                        sigma_pre_inp[j, ((N_wl-1)-k)] = 10 ** (log_sigma[0, j, z])
                            
                    # If pressure above maximum, set to value at max pressure
                    elif (x == -2):
                        sigma_pre_inp[j, ((N_wl-1)-k)] = 10 ** (log_sigma[(N_P-1), j, z])
            
                    # Interpolate sigma in logsace, then power to get interp array
                    else:
                        reduced_sigma = log_sigma[x:x+2, j, z]
                            
                        sigma_pre_inp[j, ((N_wl-1)-k)] =  10 ** (b1*(reduced_sigma[0]) +
                                                                 b2*(reduced_sigma[1]))
                            
                # Log averaging
                elif (mode == 2):
                    
                    # If pressure below minimum, set to value at min pressure
                    if (x == -1):
                        sigma_in_bin = np.mean(log_sigma[0, j, z_l:z_r+1])
                        sigma_pre_inp[j, ((N_wl-1)-k)] = 10 ** (sigma_in_bin)
                            
                    # If pressure above maximum, set to value at max pressure
                    elif (x == -2):
                        sigma_in_bin = np.mean(log_sigma[(N_P-1), j, z_l:z_r+1])
                        sigma_pre_inp[j, ((N_wl-1)-k)] = 10 ** (sigma_in_bin)
        
                    # Interpolate sigma in logsace, then power to get interp array
                    else:
                        sigma_in_bin_P1 = np.mean(log_sigma[x, j, z_l:z_r+1])
                        sigma_in_bin_P2 = np.mean(log_sigma[x+1, j, z_l:z_r+1])
                        
                        sigma_pre_inp[j, ((N_wl-1)-k)] =  10 ** (b1*(sigma_in_bin_P1) +
                                                                 b2*(sigma_in_bin_P2))
                                                
    return sigma_pre_inp

@jit(nopython = True)
def T_interpolation_init(T_grid, T):
    
    y = 0  # Index in cross secion arrays prior to fine temperature value
        
    if (T < T_grid[0]):   # If temperature falls off LHS of temperaure grid
        y = -1            # Special value (-1) stored, interpreted in interpolator
        w_T = 0.0         # Weight not used in this case
            
    elif (T >= T_grid[-1]):   # If temperature falls off RHS of temperaure grid
        y = -2                # Special value (-2) stored, interpreted in interpolator
        w_T = 0.0             # Weight not used in this case
        
    else:
            
        # Have to use prior_index (V1) here as T_grid is not uniformly spaced
        y = prior_index(T_grid, T, 0)      # Index in cross secion arrays prior to desired temperature value
            
        # Pre-computed temperature values to left and right of desired temperature value
        T1 = T_grid[y]
        T2 = T_grid[y+1]
            
        # Precompute temperature interpolation weight factor
        w_T = (1.0/((1.0/T2) - (1.0/T1)))
  
    return y, w_T

@jit(nopython = True)
def T_interpolate(N_T, N_wl, sigma_pre_inp, T_grid, T, y, w_T):
    
    sigma_inp = np.zeros(shape=(N_wl))
            
    T1 = T_grid[y]
    T2 = T_grid[y+1]
            
    for k in range(N_wl):   # Loop over wavelengths
                
        # If T_fine below min value (100 K), set sigma to value at min T
        if (y == -1):
            sigma_inp[k] = sigma_pre_inp[0, k]
                    
        # If T_fine above max value (3500 K), set sigma to value at max T
        elif (y == -2):
            sigma_inp[k] = sigma_pre_inp[(N_T-1), k]
            
        # Interpolate sigma to fine temperature grid value
        else: 
            sig_reduced = sigma_pre_inp[y:y+2, k]
            sig_1, sig_2 = sig_reduced[0], sig_reduced[1]    # sigma(T1)[i,k], sigma(T2)[i,k]
                    
            sigma_inp[k] =  (np.power(sig_1, (w_T*((1.0/T2) - (1.0/T)))) *
                             np.power(sig_2, (w_T*((1.0/T) - (1.0/T1)))))
            
    return sigma_inp


## Written by Lia
## To separate out reading step
def load_db(filename='./Opacity_database_0.01cm-1.hdf5'):

    print("Reading opacity database file")
    opac_file = h5py.File(filename, 'r')

    #***** Read in T and P grids used in opacity files*****#
    T_grid = np.array(opac_file['H2O/T'])            # H2O here simply used as dummy (same grid for all molecules)
    log_P_grid = np.array(opac_file['H2O/log(P)'])   # Units: log10(P/bar)!
    
    #***** Read in wavenumber arrays used in opacity files*****#
    nu_opac = np.array(opac_file['H2O/nu'])     # H2O here simply used as dummy (same grid for all molecules)

    return T_grid, log_P_grid, nu_opac, opac_file


# Algorithm for loading cross-sections from one pressure and temperature
# from cross-section table of a single species (log_sigma)
def _get_one_PT(log_sigma, T_grid, log_P_grid, nu_opac, 
                new_P, new_T, wl_out, opacity_treatment):
    
    if   (opacity_treatment == 'Opacity-sample'): calculation_mode = 1
    elif (opacity_treatment == 'Log-avg'):        calculation_mode = 2

    N_P = len(log_P_grid)              # No. of pressures in opacity files
    N_T = len(T_grid)                  # No. of temperatures in opacity files

    # Convert model wavelength grid to wavenumber grid
    nu_out = 1.0e4/wl_out    # Model wavenumber grid (cm^-1)
    nu_out = nu_out[::-1]    # Reverse direction, such that increases with wavenumber
    
    N_nu = len(nu_out)    # Number of wavenumbers on model grid
    N_wl = len(wl_out)    # Number of wavelengths on model grid
    
    # Initialise arrays of wavenumber locations of left and right bin edges
    #nu_l = np.zeros(N_nu)   # Left edge
    #nu_r = np.zeros(N_nu)   # Right edge
    # Look below for definitions of nu_l, nu_r
            
    # Find logarithm of desired pressure
    log_P = np.log10(new_P)

    # If pressure below minimum, do not interpolate
    if (log_P < log_P_grid[0]):
        x = -1      # Special value (1) used in opacity inialiser
        w_P = 0.0
    # If pressure above maximum, do not interpolate
    elif (log_P >= log_P_grid[-1]):
        x = -2      # Special value (2) used in opacity inialiser
        w_P = 0.0
    else:
        # Closest P indicies in opacity grid corresponding to model pressure
        x = prior_index_V2(log_P, log_P_grid[0], log_P_grid[-1], N_P)
        # Weights - fractional distance along pressure axis of sigma array
        w_P = (log_P-log_P_grid[x])/(log_P_grid[x+1]-log_P_grid[x])     
            
    # Precalculate interpolation pre-factors to reduce computation overhead
    b1 = (1.0-w_P)
    b2 = w_P  
    
    # Find wavenumber indicies in arrays of model grid
    # Vectorized by Lia
    nu_edges = np.append(nu_out[0] - (nu_out[1] - nu_out[0]), 
                         nu_out)
    nu_edges = np.append(nu_edges, nu_out[-1] + (nu_out[-1] - nu_out[-2]))
    nu_l = 0.5 * (nu_edges[:-2] + nu_edges[1:-1])
    nu_r = 0.5 * (nu_edges[1:-1] +  nu_edges[2:])
    
    '''for k in range(N_nu):
        
        if (k != 0) and (k != (N_nu-1)):    
            nu_l[k] = 0.5*(nu_out[k-1] + nu_out[k])
            nu_r[k] = 0.5*(nu_out[k] + nu_out[k+1])
        
        # Special case for boundary values
        elif (k == 0): 
            nu_l[k] = nu_out[k] - 0.5*(nu_out[k+1] - nu_out[k])
            nu_r[k] = 0.5*(nu_out[k] + nu_out[k+1])
        elif (k == (N_nu-1)):
            nu_l[k] = 0.5*(nu_out[k-1] + nu_out[k])
            nu_r[k] = nu_out[k] + 0.5*(nu_out[k] - nu_out[k-1])'''

    # Evaluate temperature interpolation weighting factor
    y, w_T = T_interpolation_init(T_grid, new_T)
    
    sigma_pre_T_inp = P_interpolate_wl_initialise(N_T, N_P, N_wl, 
                                                  log_sigma, nu_l, nu_out, nu_r, 
                                                  nu_opac, N_nu, x, b1, b2, calculation_mode)

    sigma_result = T_interpolate(N_T, N_wl, sigma_pre_T_inp, T_grid, new_T, y, w_T)

    return sigma_result


def Extract_opacity(chemical_species, P, T, wl_out, opacity_treatment):
    
    '''Convienient function to read in all opacities and pre-interpolate
       them onto the desired pressure, temperature, and wavelength grid'''

    T_grid, log_P_grid, nu_opac, opac_file = load_db()

    # Initialise molecular and atomic opacity array, interpolated to model wavelength grid
    #sigma_stored = np.zeros(shape=(N_species, N_wl))
    # Lia -- Making this a dictionary instead of a numpy array. It's easier to use for plotting.
    # Later, a 2D numpy array will be faster for summing across a large number of species
    # But for now, two dozen is not a lot.
    sigma_stored = dict()
    
    #***** Process molecular and atomic opacities *****#
    
    # Load molecular and atomic absorption cross sections
    for q in chemical_species:
        
        log_sigma = np.array(opac_file[q + '/log(sigma)']).astype(np.float64)

        sigma_stored[q] = _get_one_PT(log_sigma, T_grid, log_P_grid, nu_opac,
                                              P, T, wl_out, opacity_treatment)
        
        del log_sigma   # Clear raw cross section to free up memory
        
        print(q + " done")
    
    opac_file.close()
    
    return sigma_stored

# Written by Lia for a set of (p,T) values
def Extract_opacity_PTpairs(chemical_species, P, T, wl_out, opacity_treatment):
    
    '''Convienient function to read in all opacities and pre-interpolate
       them onto the desired pressure, temperature, and wavelength grid'''

    assert len(P) == len(T)

    T_grid, log_P_grid, nu_opac, opac_file = load_db()

    # Initialise molecular and atomic opacity array, interpolated to model wavelength grid
    #sigma_stored = np.zeros(shape=(N_species, N_wl))
    # Lia -- Making this a dictionary instead of a numpy array. It's easier to use for plotting.
    # Later, a 2D numpy array will be faster for summing across a large number of species
    # But for now, two dozen is not a lot.
    sigma_stored = dict()
    
    #***** Process molecular and atomic opacities *****#
    
    # Load molecular and atomic absorption cross sections
    for q in chemical_species:
        
        log_sigma = np.array(opac_file[q + '/log(sigma)']).astype(np.float64)
        
        result = np.zeros(shape=(len(P), len(wl_out)))
        for i in range(len(P)):
             result[i,:] = _get_one_PT(log_sigma, T_grid, log_P_grid, nu_opac,
                                       P[i], T[i], wl_out, opacity_treatment)
        
        sigma_stored[q] = result
        
        del log_sigma   # Clear raw cross section to free up memory
        
        print(q + " done")
    
    opac_file.close()
    
    return sigma_stored


def plot_opacity(chemical_species, sigma_stored, P, T, wl_grid, savefig=False, **kwargs):
    
    # Max number of species this can plot is 9 (clustered beyond that!)
    
    # Optional smoothing of cross sections (can improve clarity)
    smooth = False
    smooth_factor = 5
    
    # Specify cross sections to plot, along with colours for each
    #colours_plot = np.array(['royalblue', 'purple', 'crimson', 'orange', 'black', 'grey', 'green', 'magenta', 'chocolate'])
    
    # Initialise plot
    #ax = plt.gca()
    #ax.set_xscale("log")
    
    ax = plt.subplot(111)
    #xmajorLocator   = MultipleLocator(1.0)
    #xmajorFormatter = FormatStrFormatter('%.1f')
    #xminorLocator   = MultipleLocator(0.2)
    
    #ax.xaxis.set_major_locator(xmajorLocator)
    #ax.xaxis.set_major_formatter(xmajorFormatter)
    #ax.xaxis.set_minor_locator(xminorLocator)
    
    # Plot each cross section
    for species in chemical_species:
        #species_idx = np.where(chemical_species == species)[0][0]
        sigma_plt = sigma_stored[species]*1.0e4   # Cross section of species q at given (P,T) pair (cm^2)
        
        if (smooth == True):
            sigma_plt = gauss_conv(sigma_plt, sigma=smooth_factor, mode='nearest')
            
        # Plot cross section
        plt.semilogy(wl_grid, sigma_plt, label=species, **kwargs)
    
    plt.ylim([1.0e-28, 2.0e-18])
    plt.xlim([min(wl_grid), max(wl_grid)])
    plt.ylabel(r'$\mathrm{Cross \, \, Section \, \, (cm^{2})}$', fontsize = 15)
    plt.xlabel(r'$\mathrm{Wavelength} \; \mathrm{(\mu m)}$', fontsize = 15)
    
    ax.text(min(wl_grid)*1.05, 6.0e-19, (r'$\mathrm{T = }$' + str(T) + \
                                         r'$\mathrm{K \, \, P = }$' + \
                                         str(P*1000) + r'$\mathrm{mbar}$'), fontsize = 14)
    
    legend = plt.legend(loc='upper right', frameon=False, prop={'size':6}, ncol=2)
    
    '''for legline in legend.legendHandles:
    legline.set_linewidth(1.0)'''

    plt.show()

    if savefig:
        plt.savefig('./cross_sections_' + str(T) + 'K_' + str(P*1000) + 'mbar.pdf', bbox_inches='tight', fmt='pdf', dpi=1000)

    plt.close()
    
    return

#***** Begin main program ***** 

# This code will execute if you run the script from the terminal
# Otherwise, no
if __name__ == '__main__':
    # Specify which molecules you want to extract from the database (full list available in the readme)
    chemical_species = np.array(['H2O', 'CH4', 'NH3', 'HCN', 'CO', 'CO2'])
    
    # At what temperature and pressure do you desire the cross sections?
    P = 1.0e-3    # Pressure (bar)
    T = 1000.0    # Temperature (K)        

    # Specify wavelength grid to extract cross section onto
    wl_min = 0.4  # Minimum wavelength of grid (micron)
    wl_max = 5.0  # Maximum wavelength of grid (micron)
    N_wl = 1000   # Number of wavelength points

    wl = np.linspace(wl_min, wl_max, N_wl)  # Uniform grid used here for demonstration purposes   
    
    # Either sample the nearest wavelength points from the high resolution (R~10^6) cross section database or use an averaging prescription 
    opacity_treatment = 'Log-avg'           # Options: Opacity-sample / Log-avg
    #opacity_treatment = 'Opacity-sample'   # Opacity sampling is faster, but for low-resolution wavelength grids log averaging is recommended
    
    # Extract desired cross sections from the database
    cross_sections = Extract_opacity(chemical_species, P, T, wl, opacity_treatment)   # Format: np array(N_species, N_wl) / Units: (m^2 / species)
    
    # Example: seperate H2O cross section, and print to terminal
    H2O_cross_section = cross_sections['H2O']    # Format: np array(N_wl) / Units: (m^2 / molecule)
    #print (H2O_cross_section)
    
    # Plot cross sections
    plot_opacity(chemical_species, cross_sections, P, T, wl, savefig=True, alpha=0.8, lw=0.5)

