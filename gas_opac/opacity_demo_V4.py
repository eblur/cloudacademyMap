# Demostrates how to open the opacity database and interpolate to given (P,T) and wavelength grid
# Author: Ryan J. MacDonald
# V1.0: Evaluates cross sections at a single P,T point (8th November 2018)
# V2.0: Evaluates cross sections on a grid of (P,T) points (6th December 2018)
# V3.0: Evaluates collisionally-induced opacity on a grid of (T) points (17th February 2019)
# V4.0: Inclusion of H- bound-free and free-free opacities (26th February 2019)

import numpy as np
import h5py
from scipy.ndimage import gaussian_filter1d as gauss_conv
from numba.decorators import jit
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FormatStrFormatter, NullFormatter

from pylab import rcParams
plt.style.use('classic')
plt.rc('font', family='serif')

#***** Begin function declarations *****# 

@jit(nopython = True)
def prior_index(vec, value, start):
    
    '''Finds the previous index of a grid closest to a specified value
    '''
    
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
    
    '''Finds the previous index of a UNIFORM grid closest to a specified value.
    
       A uniform grid dramatically speeds calculation over a non-uniform grid.
    '''
    
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
    
       A uniform grid dramatically speeds calculation over a non-uniform grid.
    '''
    
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
def P_interpolate_wl_initialise(nu_grid, N_P_out, N_T, N_P, N_wl_out, N_nu_out, 
                                log_sigma, x, z_l, z, z_r, b1, b2, mode):
    
    '''Interpolates raw cross sections onto the output P and wl grids.
       
       Note: input sigma has format log10(cross_sec)[log(P)_grid, T_grid, nu_grid], 
             whilst output has format cross_sec[log(P)_out, T_grid, wl_out]
             
       The input is in wavenumnber to take advantage of fast prior index location 
       on a uniform grid, which wouldn't work for the (non-uniform) wavelength grid
       Array reversal to output in increasing wavelength is handled by indexing
       by a factor of (N_wl-1)-k throughout. 
    '''
    
    sigma_pre_inp = np.zeros(shape=(N_P_out, N_T, N_wl_out))
    
    N_nu = len(nu_grid)   # Number of wavenumber points in sigma array
        
    for i in xrange(N_P_out):            # For each pressure in output array
        for j in xrange(N_T):            # For each temperature in input array
            for k in xrange(N_nu_out):   # Note that the k here is looping over wavenumber
                            
                # If nu (wl) point out of range of opacity grid, set opacity to zero
                if ((z[k] == 0) or (z[k] == (N_nu-1))):
                    sigma_pre_inp[i, j, ((N_wl_out-1)-k)] = 0.0
                
                else:
                    
                    # Opacity sampling
                    if (mode == 1):
                    
                        # If pressure below minimum, set to value at min pressure
                        if (x[i] == -1):
                            sigma_pre_inp[i, j, ((N_wl_out-1)-k)] = 10 ** (log_sigma[0, j, z[k]])
                            
                        # If pressure above maximum, set to value at max pressure
                        elif (x[i] == -2):
                            sigma_pre_inp[i, j, ((N_wl_out-1)-k)] = 10 ** (log_sigma[(N_P-1), j, z[k]])
            
                        # Interpolate sigma in logsace, then power to get interp array
                        else:
                            reduced_sigma = log_sigma[x[i]:x[i]+2, j, z[k]]
                            
                            sigma_pre_inp[i, j, ((N_wl_out-1)-k)] =  10 ** (b1[i]*(reduced_sigma[0]) +
                                                                            b2[i]*(reduced_sigma[1]))
                            
                    # Log averaging
                    elif (mode == 2):
                    
                        # If pressure below minimum, set to value at min pressure
                        if (x[i] == -1):
                            sigma_in_bin = np.mean(log_sigma[0, j, z_l[k]:z_r[k]+1])
                            sigma_pre_inp[i, j, ((N_wl_out-1)-k)] = 10 ** (sigma_in_bin)
                            
                        # If pressure above maximum, set to value at max pressure
                        elif (x[i] == -2):
                            sigma_in_bin = np.mean(log_sigma[(N_P-1), j, z_l[k]:z_r[k]+1])
                            sigma_pre_inp[i, j, ((N_wl_out-1)-k)] = 10 ** (sigma_in_bin)
        
                        # Interpolate sigma in logsace, then power to get interp array
                        else:
                            sigma_in_bin_P1 = np.mean(log_sigma[x[i], j, z_l[k]:z_r[k]+1])
                            sigma_in_bin_P2 = np.mean(log_sigma[x[i]+1, j, z_l[k]:z_r[k]+1])
                        
                            sigma_pre_inp[i, j, ((N_wl_out-1)-k)] =  10 ** (b1[i]*(sigma_in_bin_P1) +
                                                                            b2[i]*(sigma_in_bin_P2))
                    
    return sigma_pre_inp

@jit(nopython = True)
def wl_initialise_cia(nu_cia, N_T_cia, N_wl_out, N_nu_out, log_cia, nu_l, nu_out, nu_r, mode):
    
    ''' Interpolates raw collisionally-induced absorption (CIA) binary cross 
        section onto the desired model wl grid.
       
        Input cia has format log10(alpha)[T_grid, nu_grid],
        whilst output has format alpha[T_grid, wl_model].
              
        The input is in wavenumnber to take advantage of fast prior index 
        location on a uniform grid, which wouldn't work for the (non-uniform) 
        wavelength grid. Array reversal to output in increasing wavelength is 
        handled by indexing by a factor of (N_wl-1)-k throughout .
              
        Wavelength initialisation is handled via either opacity sampling
        (choosing nearest pre-computed wavelength point) or via averaging
        the logarithm of the cross section over the wavelength bin range
        surrounding each wavelength on the output model wavelength grid.
       
    '''
    
    cia_pre_inp = np.zeros(shape=(N_T_cia, N_wl_out))
    
    N_nu_cia = len(nu_cia)   # Number of wavenumber points in CIA array

    for i in xrange(N_T_cia):  
        for k in xrange(N_nu_out):
            
            z_l = closest_index(nu_l[k], nu_cia[0], nu_cia[-1], N_nu_cia)
            z = closest_index(nu_out[k], nu_cia[0], nu_cia[-1], N_nu_cia)
            z_r = closest_index(nu_r[k], nu_cia[0], nu_cia[-1], N_nu_cia)
            
            # If wl out of range of opacity, set opacity to zero
            if ((z_l == 0) or (z_r == (N_nu_cia-1))):
                cia_pre_inp[i, ((N_wl_out-1)-k)] = 0.0
                
            else:
                
                # Opacity sampling
                if (mode == 1):
                    
                    cia_pre_inp[i, ((N_wl_out-1)-k)] = 10 ** (log_cia[i, z])
                    
                # Log averaging
                elif (mode == 2):
                    
                    cia_in_bin = np.mean(log_cia[i, z_l:z_r+1])
                    cia_pre_inp[i, ((N_wl_out-1)-k)] = 10 ** (cia_in_bin)
               
    return cia_pre_inp

@jit(nopython = True)
def T_interpolation_init(N_T_out, T_grid, T_out, y):
    
    ''' Precomputes the T interpolation weight factors, so this does not
        need to be done multiple times across all species.
    '''
    
    w_T = np.zeros(N_T_out)  # Temperature interpolation weight factors
    
    # Find T index in cross secion arrays prior to fine temperature value
    for j in xrange(N_T_out):
        
        if (T_out[j] < T_grid[0]):    # If fine temperature point falls off LHS of temperaure grid
            y[j] = -1                 # Special value (-1) stored, interpreted in interpolator
            w_T[j] = 0.0              # Weight not used in this case
            
        elif (T_out[j] >= T_grid[-1]):    # If fine temperature point falls off RHS of temperaure grid
            y[j] = -2                     # Special value (-2) stored, interpreted in interpolator
            w_T[j] = 0.0                  # Weight not used in this case
            
        else:
                
            # Have to use prior_index (V1) here as T_grid is not uniformly spaced
            y[j] = prior_index(T_grid, T_out[j], 0)     # Index in cross secion arrays prior to desired temperature value
                
            # Pre-computed temperature values to left and right of desired temperature value
            T1 = T_grid[y[j]]
            T2 = T_grid[y[j]+1]
                
            # Precompute temperature interpolation weight factor
            w_T[j] = (1.0/((1.0/T2) - (1.0/T1)))
  
    return w_T

@jit(nopython = True)
def T_interpolate_sigma(N_P_out, N_T_out, N_T, N_wl_out, sigma_pre_inp, T_grid, T_out, y, w_T):
    
    '''Interpolates pre-processed cross section onto the output T grid.
       
       Note: input sigma has format log10(cross_sec)[log(P)_out, T_grid, wl_out], 
             whilst output has format cross_sec[log(P)_out, T_out, wl_out].
             
       Output is the final interpolated cross section as a 3D array on the
       desired P_out, T_out, wl_out grids.
    '''
    
    sigma_inp = np.zeros(shape=(N_P_out, N_T_out, N_wl_out))
    
    for i in xrange(N_P_out):       # Loop over output pressure array
        for j in xrange(N_T_out):   # Loop over output temperature array
            
            T = T_out[j]            # Temperature we wish to interpolate to
            T1 = T_grid[y[j]]
            T2 = T_grid[y[j]+1]
            
            for k in xrange(N_wl_out):  # Loop over wavelengths
                
                # If T_out below min pre-computed value (100 K), set sigma to value at min T
                if (y[j] == -1):
                    sigma_inp[i, j, k] = sigma_pre_inp[i, 0, k]
                    
                # If T_out above max pre-computed value (3500 K), set sigma to value at max T
                elif (y[j] == -2):
                    sigma_inp[i, j, k] = sigma_pre_inp[i, (N_T-1), k]
            
                # Interpolate sigma to output temperature grid value
                else: 
                    sig_reduced = sigma_pre_inp[i, y[j]:y[j]+2, k]
                    sig_1, sig_2 = sig_reduced[0], sig_reduced[1]    # sigma(T1)[i,k], sigma(T2)[i,k]
                    
                    sigma_inp[i, j, k] =  (np.power(sig_1, (w_T[j]*((1.0/T2) - (1.0/T)))) *
                                           np.power(sig_2, (w_T[j]*((1.0/T) - (1.0/T1)))))
            
    return sigma_inp

@jit(nopython = True)
def T_interpolate_cia(N_T_out, N_T_cia, N_wl_out, cia_pre_inp, T_grid_cia, T_out, y, w_T):
    
    ''' Interpolates pre-processed collisionally-induced absorption (CIA) 
        binary cross section onto the output T grid.
       
        Note: input sigma has format alpha[T_grid, wl_model], 
              whilst output has format alpha[T_fine, wl_model].
             
        Output is the interpolated cia cross section as a 2D array on the
        desired T_out, wl_out grids.
        
    '''
    
    cia_inp = np.zeros(shape=(N_T_out, N_wl_out))

    for j in xrange(N_T_out):   # Loop over output temperature array
            
        T = T_out[j]            # Temperature we wish to interpolate to
        T1 = T_grid_cia[y[j]]
        T2 = T_grid_cia[y[j]+1]
            
        for k in xrange(N_wl):   # Loop over wavelengths
                
            # If T_fine below min value (200 K), set cia to value at min T
            if (y[j] == -1):
                cia_inp[j, k] = cia_pre_inp[0, k]
                
            # If T_fine above max value (3500 K), set cia to value at max T
            elif (y[j] == -2):
                cia_inp[j, k] = cia_pre_inp[(N_T_cia-1), k]
            
            # Interpolate sigma to fine temperature grid value
            else: 
                cia_reduced = cia_pre_inp[y[j]:y[j]+2, k]
                cia_1, cia_2 = cia_reduced[0], cia_reduced[1]    # sigma(T1)[i,k], sigma(T2)[i,k]
                    
                cia_inp[j, k] =  (np.power(cia_1, (w_T[j]*((1.0/T2) - (1.0/T)))) *
                                  np.power(cia_2, (w_T[j]*((1.0/T) - (1.0/T1)))))
            
    return cia_inp

def Extract_opacity(chemical_species, cia_pairs, P_out, T_out, wl_out, opacity_treatment):
    
    '''Main function to read in all opacities and interpolate them onto the 
       desired pressure, temperature, and wavelength grids.
    '''
    
    print("Now reading in cross sections")
    
    # First, check from config.py which opacity calculation mode is specified
    if   (opacity_treatment == 'Opacity-sample'): calculation_mode = 1
    elif (opacity_treatment == 'Log-avg'):        calculation_mode = 2

    #***** Firstly, we initialise the various quantities needed for pre-interpolation*****#
    
    # Open HDF5 files containing molecular + atomic opacities and CIA
    opac_file = h5py.File('./Opacity_database_0.01cm-1.hdf5', 'r')
    cia_file = h5py.File('./Opacity_database_cia.hdf5', 'r')
    
    #***** Read in T and P grids used in opacity files*****#
    T_grid = np.array(opac_file['H2O/T'])            # H2O here simply used as dummy (same grid for all molecules)
    log_P_grid = np.array(opac_file['H2O/log(P)'])   # Units: log10(P/bar)
    log_P_out = np.log10(P_out)                      # Units: log10(P/bar)
    
    #***** Read in wavenumber arrays used in opacity files*****#
    nu_grid = np.array(opac_file['H2O/nu'])     # H2O here simply used as dummy (same grid for all molecules)

    # Convert model wavelength grid to wavenumber grid
    nu_out = 1.0e4/wl_out    # Model wavenumber grid (cm^-1)
    nu_out = nu_out[::-1]    # Reverse direction, such that increases with wavenumber
    
    N_nu = len(nu_grid)                # No. of wavenumbers in each opacity file (2480001)
    N_P = len(log_P_grid)              # No. of pressures in opacity files
    N_T = len(T_grid)                  # No. of temperatures in opacity files
    N_species = len(chemical_species)  # No. of chemical species user wishes to store
    N_cia_pairs = len(cia_pairs)       # Number of CIA pairs user wishes to store (= 2 for H2-H2 & H2-He)
    
    N_P_out = len(P_out)      # No. of pressures on output pressure grid
    N_T_out = len(T_out)      # No. of temperatures on output temperature grid
    N_nu_out = len(nu_out)    # No. of wavenumbers on output grid
    N_wl_out = len(wl_out)    # No. of wavelengths on output grid
    
    # Initialise arrays of wavenumber locations of left and right bin edges
    nu_l = np.zeros(N_nu_out)   # Left edge
    nu_r = np.zeros(N_nu_out)   # Right edge
    
    # Initialise arrays of indicies on wavenumber opacity grid corresponding to nu_l, nu_out, and nu_r
    z_l = np.zeros(N_wl_out, dtype=np.int)   # Bin left edge
    z = np.zeros(N_wl_out, dtype=np.int)     # Bin centre
    z_r = np.zeros(N_wl_out, dtype=np.int)   # Bin right edge
    
    # Initialise array of indicies on pre-calculated pressure opacity grid prior to defined atmosphere layer pressures
    x = np.zeros(N_P_out, dtype=np.int)
    
    # Weights
    w_P = np.zeros(N_P_out)
            
    # Useful functions of weights for interpolation
    b1 = np.zeros(shape=(N_P_out))
    b2 = np.zeros(shape=(N_P_out))
    
    # Now find closest P indicies in opacity grid corresponding to output pressures
    for i in xrange(N_P_out):
        
        # If pressure below minimum, do not interpolate
        if (log_P_out[i] < log_P_grid[0]):
            x[i] = -1      # Special value (1) used in opacity initialiser
            w_P[i] = 0.0
        
        # If pressure above maximum, do not interpolate
        elif (log_P_out[i] >= log_P_grid[-1]):
            x[i] = -2      # Special value (2) used in opacity initialiser
            w_P[i] = 0.0
        
        else:
            x[i] = prior_index_V2(log_P_out[i], log_P_grid[0], log_P_grid[-1], N_P)
        
            # Weights - fractional distance along pressure axis of cross section array
            w_P[i] = (log_P_out[i]-log_P_grid[x[i]])/(log_P_grid[x[i]+1]-log_P_grid[x[i]])     
         
        # Precalculate interpolation pre-factors to reduce computation overhead
        b1[i] = (1.0-w_P[i])
        b2[i] = w_P[i]  
            
    # Find wavenumber indicies on input grid closest to values on output grid
    for k in xrange(N_nu_out):
        
        if (k != 0) and (k != (N_nu_out-1)):    
            nu_l[k] = 0.5*(nu_out[k-1] + nu_out[k])
            nu_r[k] = 0.5*(nu_out[k] + nu_out[k+1])
        
        # Boundary values of each output wavenumber point
        elif (k == 0): 
            nu_l[k] = nu_out[k] - 0.5*(nu_out[k+1] - nu_out[k])
            nu_r[k] = 0.5*(nu_out[k] + nu_out[k+1])
        elif (k == (N_nu-1)):
            nu_l[k] = 0.5*(nu_out[k-1] + nu_out[k])
            nu_r[k] = nu_out[k] + 0.5*(nu_out[k] - nu_out[k-1])
        
        # Calculate closest indicies to output wavenumber bin left, centre, and right edge
        z_l[k] = closest_index(nu_l[k], nu_grid[0], nu_grid[-1], N_nu)
        z[k] = closest_index(nu_out[k], nu_grid[0], nu_grid[-1], N_nu)
        z_r[k] = closest_index(nu_r[k], nu_grid[0], nu_grid[-1], N_nu)
        
    # Initialise molecular and atomic opacity array, interpolated to output (P,T,wl) grid
    sigma_stored = np.zeros(shape=(N_species, N_P_out, N_T_out, N_wl_out))
    
    # Initialise CIA opacity array, interpolated to to output (T,wl) grid
    cia_stored = np.zeros(shape=(N_cia_pairs, N_T_out, N_wl))
    
    #***** Process collisionally Induced Absorption (CIA) *****#
    
    for q in xrange(N_cia_pairs):
        
        cia_pair_q = cia_pairs[q]     # CIA pair name
        
        #***** Read in T grid used in this CIA file *****#
        T_grid_cia_q = np.array(cia_file[cia_pair_q + '/T'])
        N_T_cia_q = len(T_grid_cia_q)  # Number of temperatures on this grid
        
        # Read in wavenumber array used in this CIA file
        nu_cia_q = np.array(cia_file[cia_pair_q + '/nu'])
    
    # Evaluate temperature interpolation weighting factor
        y_cia = np.zeros(N_T_out, dtype=np.int)
        w_T_cia = T_interpolation_init(N_T_out, T_grid_cia_q, T_out, y_cia)   
        
        # Read in log10(binary cross section) for specified cia pair
        log_cia = np.array(cia_file[cia_pair_q + '/log(cia)'])  
        
        # Pre-interpolate cross section to desired wl grid 
        cia_pre_inp = wl_initialise_cia(nu_cia_q, N_T_cia_q, N_wl_out, N_nu_out, log_cia, nu_l, nu_out, nu_r, calculation_mode)  
        
        # Interplate to desired temperature
        cia_stored[q,:,:] = T_interpolate_cia(N_T_out, N_T_cia_q, N_wl_out, cia_pre_inp, T_grid_cia_q, T_out, y_cia, w_T_cia)     

        # Clear raw cross section to free up memory
        del log_cia, cia_pre_inp, nu_cia_q, w_T_cia, y_cia  
        
        print(cia_pair_q + " done")
        
    #***** Process molecular and atomic opacities *****#
    
    # Evaluate temperature interpolation weighting factor
    y_sigma = np.zeros(N_T_out, dtype=np.int)
    w_T_sigma = T_interpolation_init(N_T_out, T_grid, T_out, y_sigma)
    
    # Load molecular and atomic absorption cross sections
    for q in xrange(N_species):
            
        species_q = chemical_species[q]    # Species name specified by user
        
        # Read in log10(cross section) of specified species
        log_sigma = np.array(opac_file[species_q + '/log(sigma)']).astype(np.float64)      
            
        # Pre-interpolate cross section to desired P and wl grid 
        sigma_pre_inp = P_interpolate_wl_initialise(nu_grid, N_P_out, N_T, N_P, N_wl_out, N_nu_out, log_sigma, x, z_l, z, z_r, b1, b2, calculation_mode)
                
        # Interplate to desired temperature
        sigma_stored[q,:,:,:] = T_interpolate_sigma(N_P_out, N_T_out, N_T, N_wl_out, sigma_pre_inp, T_grid, T_out, y_sigma, w_T_sigma)
        
        # Clear raw cross section to free up memory
        del log_sigma, sigma_pre_inp   
        
        print(species_q + " done")
    
    # Clear up storage
    del z, z_l, z_r, nu_grid, nu_l, nu_r, nu_out, y_sigma, w_T_sigma
    
    opac_file.close()
    
    return sigma_stored, cia_stored

@jit
def H_minus_bound_free(wl_um):
    
    ''' Computes the bound-free cross section (alpha_bf) of the H- ion as a 
        function of wavelength. The fitting function is taken from "The 
        Observation and Analysis of Stellar Photospheres" (Gray, 2005).
        
        The extinction coefficient (in m^-1) can then be calculated via:
        alpha_bf * n_(H-) [i.e. multiply by the H- number density (in m^-3) ].
    
        Inputs:
           
        wl_um => array of wavelength values (um)
       
        Outputs:
           
        alpha_bf => bound-free H- cross section (m^2 / n_(H-) ) at each input
                    wavelength
        
    '''
    
    # Initialise array to store absorption coefficients
    alpha_bf = np.zeros(shape=(len(wl_um)))
    
    # Convert micron to A (as used in bound-free fit)
    wl_A = wl_um * 1.0e4  
    
    # Fitting function constants (Gray, 2005, p.156)
    a0 = 1.99654
    a1 = -1.18267e-5
    a2 = 2.64243e-6
    a3 = -4.40524e-10
    a4 = 3.23992e-14
    a5 = -1.39568e-18
    a6 = 2.78701e-23
    
    for k, wl in enumerate(wl_A):
        
        # Photodissociation only possible for photons with wl < 1.6421 micron 
        if (wl <= 16421.0):
            
            # Compute bound-free absorption coefficient at this wavelength
            alpha_bf_wl = (a0 + a1*wl + a2*(wl**2) + a3*(wl**3) + a4*(wl**4) + a5*(wl**5) + a6*(wl**6)) * 1.0e-18
            alpha_bf[k] = alpha_bf_wl * 1.0e-4  # Convert from cm^2 / H- ion to m^2 / H- ion
        
        else:
            
            alpha_bf[k] = 1.0e-250   # Small value (proxy for zero, but avoids log(0) issues)
    
    return alpha_bf

@jit
def H_minus_free_free(wl_um, T_arr):
    
    ''' Computes the free-free cross section (alpha_ff) of the H- ion as a 
        function of wavelength. The fitting function is taken from "Continuous
        absorption by the negative hydrogen ion reconsidered" (John, 1987).
        
        The extinction coefficient (in m^-1) can then be calculated via:
        alpha_ff * n_H * n_(e-) [i.e. multiply by the H and e- number densities
        (both in in m^-3) ].
    
        Inputs:
           
        wl_um => array of wavelength values (um)
        T_arr => array of temperatures (K)
       
        Outputs:
           
        alpha_ff => free-free H- cross section (m^5 / n_H / n_e-) for each 
                    input wavelength and temperature
        
    '''
    
    # Initialise array to store absorption coefficients
    alpha_ff = np.zeros(shape=(len(wl_um), len(T_arr)))
        
    for i, T in enumerate(T_arr):
            
        theta = 5040.0/T      # Reciprocal temperature commonly used in these fits
        kT = 1.38066e-16 * T  # Boltzmann constant * temperature (erg)
    
        for k, wl in enumerate(wl_um):
                
            # Range of validity of fit (wl > 0.182 um)
            if (wl < 0.182):
                    
                alpha_ff[k,i] = 1.0e-250   # Small value (proxy for zero, but avoids log(0) issues)
            
            # Short wavelength fit
            elif ((wl >= 0.182) and (wl < 0.3645)):
                    
                A = np.array([518.1021, 473.2636, -482.2089, 115.5291, 0.0, 0.0])
                B = np.array([-734.8666, 1443.4137, -737.1616, 169.6374, 0.0, 0.0])
                C = np.array([1021.1775, -1977.3395, 1096.8827, -245.6490, 0.0, 0.0])
                D = np.array([-479.0721, 922.3575, -521.1341, 114.2430, 0.0, 0.0])
                E = np.array([93.1373, -178.9275, 101.7963, -21.9972, 0.0, 0.0])
                F = np.array([-6.4285, 12.3600, -7.0571, 1.5097, 0.0, 0.0])
                    
                # For each set of fit coefficients
                for n in range(1,7):
                
                    # Compute free-free absorption coefficient at this wavelength and temperature
                    alpha_ff[k,i] += 1.0e-29 * (np.power(theta, ((n + 1.0)/2.0)) * 
                                                (A[n-1]*(wl**2) + B[n-1] + C[n-1]/wl + 
                                                 D[n-1]/(wl**2) + E[n-1]/(wl**3) + 
                                                 F[n-1]/(wl**4))) * kT
                    
                alpha_ff[k,i] *= 1.0e-10  # Convert from cm^5 / H- ion to m^5 / H- ion
                    
            # Long wavelength fit
            elif (wl >= 0.3645):
                    
                A = np.array([0.0, 2483.3460, -3449.8890, 2200.0400, -696.2710, 88.2830])
                B = np.array([0.0, 285.8270, -1158.3820, 2427.7190, -1841.4000, 444.5170])
                C = np.array([0.0, -2054.2910, 8746.5230, -13651.1050, 8624.9700, -1863.8640])
                D = np.array([0.0, 2827.7760, -11485.6320, 16755.5240, -10051.5300, 2095.2880])
                E = np.array([0.0, -1341.5370, 5303.6090, -7510.4940, 4400.0670, -901.7880])
                F = np.array([0.0, 208.9520, -812.9390, 1132.7380, -655.0200, 132.9850])
                    
                # For each set of fit coefficients
                for n in range(1,7):
                
                    # Compute free-free absorption coefficient at this wavelength and temperature
                    alpha_ff[k,i] += 1.0e-29 * (np.power(theta, ((n + 1.0)/2.0)) * 
                                                (A[n-1]*(wl**2) + B[n-1] + C[n-1]/wl + 
                                                 D[n-1]/(wl**2) + E[n-1]/(wl**3) + 
                                                 F[n-1]/(wl**4))) * kT  # cm^5 / H / e-
                alpha_ff[k,i] *= 1.0e-10  # Convert from cm^5 / H- ion to m^5 / H- ion
    
    return alpha_ff

def plot_opacity(chemical_species, cia_pairs, sigma_stored, cia_stored, 
                 alpha_bf, alpha_ff, P_desired, T_desired, wl_grid):
    
    # Max number of species this can plot is 9 (cluttered beyond that!)
    
    # Optional smoothing of cross sections (can improve clarity)
    smooth = False
    smooth_factor = 5
    
    # Specify cross sections to plot, along with colours for each
    colours_sigma = np.array(['royalblue', 'purple', 'crimson', 'orange', 'black', 'grey', 'green', 'magenta', 'chocolate'])
    
    # Initialise plot
    ax = plt.gca()    
    #ax.set_xscale("log")
    
    xmajorLocator   = MultipleLocator(1.0)
    xmajorFormatter = FormatStrFormatter('%.1f')
    xminorLocator   = MultipleLocator(0.2)
    xminorFormatter = NullFormatter()
    
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.xaxis.set_minor_formatter(xminorFormatter)
    
    # Plot each cross section
    for q in range(len(chemical_species)):
        
        species = chemical_species[q]  # Species to plot cross section of 
        colour = colours_sigma[q]      # Colour of cross section for plot
        
        species_idx = np.where(chemical_species == species)[0][0]

        sigma_plt = sigma_stored[species_idx,:]   # Cross section of species q at given (P,T) pair (m^2)
        
        if (smooth == True):
            sigma_plt = gauss_conv(sigma_plt, sigma=smooth_factor, mode='nearest')
        
        # Plot cross section
        plt.semilogy(wl_grid, sigma_plt, lw=0.5, alpha = 1.0, color= colour, label = species)
    
    plt.ylim([1.0e-31, 1.0e-21])
    plt.xlim([min(wl_grid), max(wl_grid)])
    plt.ylabel(r'$\mathrm{Cross \, \, Section \, \, (m^{2} \, mol^{-1})}$', fontsize = 15)
    plt.xlabel(r'$\mathrm{Wavelength} \; \mathrm{(\mu m)}$', fontsize = 15)
    
    ax.set_xticks([1, 2, 3, 4, 5])
    
    ax.text(min(wl_grid)*1.20, 2.0e-22, (r'$\mathrm{T = }$' + str(T_desired) + r'$\mathrm{K \, \, P = }$' + str(P_desired*1000) + r'$\mathrm{mbar}$'), fontsize = 14)
    
    legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':6}, ncol=2)
    
    for legline in legend.legendHandles:
        legline.set_linewidth(1.0)

    plt.savefig('./cross_sections_' + str(T_desired) + 'K_' + str(P_desired*1000) + 'mbar.pdf', bbox_inches='tight', fmt='pdf', dpi=1000)

    plt.show()
    
    plt.clf()
    
    #***** Now plot CIA *****
    
    colours_cia = np.array(['royalblue', 'crimson'])
    
    ax = plt.gca()    
    #ax.set_xscale("log")
    
    xmajorLocator   = MultipleLocator(1.0)
    xmajorFormatter = FormatStrFormatter('%.1f')
    xminorLocator   = MultipleLocator(0.2)
    
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.xaxis.set_minor_locator(xminorLocator)
    
    # Plot each cross section
    for q in range(len(cia_pairs)):
        
        cia_pair = cia_pairs[q]  # Species to plot cross section of 
        colour = colours_cia[q]   # Colour of cross section for plot
        
        cia_idx = np.where(cia_pairs == cia_pair)[0][0]

        cia_plt = cia_stored[cia_idx,:]  # Cross section of species q at given (P,T) pair (m^5)
        
        # Plot cross section
        plt.semilogy(wl_grid, cia_plt, lw=0.5, alpha = 1.0, color = colour, label = cia_pair)
    
    plt.ylim([1.0e-60, 1.0e-52])
    plt.xlim([min(wl_grid), max(wl_grid)])
    plt.ylabel(r'$\mathrm{Binary \, \, Cross \, \, Section \, (CIA) \, \, (m^{5} mol^{-2})}$', fontsize = 15)
    plt.xlabel(r'$\mathrm{Wavelength} \; \mathrm{(\mu m)}$', fontsize = 15)
    
    ax.text(min(wl_grid)*1.20, 3.0e-53, (r'$\mathrm{T = }$' + str(T_desired) + r'$\mathrm{K}$'), fontsize = 14)
    
    legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':8})
    
    for legline in legend.legendHandles:
        legline.set_linewidth(1.0)

    plt.savefig('./CIA_cross_sections_' + str(T_desired) + 'K.pdf', bbox_inches='tight', fmt='pdf', dpi=1000)
    
    plt.show()
    
    plt.clf()
    
    #***** Now plot H- bound-free opacity *****
        
    # Initialise plot
    ax = plt.gca()    
        
    xmajorLocator   = MultipleLocator(0.5)
    xmajorFormatter = FormatStrFormatter('%.1f')
    xminorLocator   = MultipleLocator(0.1)
    xminorFormatter = NullFormatter()
        
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.xaxis.set_minor_formatter(xminorFormatter)
    
    # Plot cross section
    plt.semilogy(wl, alpha_bf, lw=1.0, alpha = 0.5, color= 'crimson', label = r'H- bound-free')
    
    plt.ylim([1.0e-23, 1.0e-20])
    plt.xlim([min(wl), 1.8])
    plt.ylabel(r'$\mathrm{Cross \, \, Section \, \, (m^{2} \, / n_{H-})}$', fontsize = 15)
    plt.xlabel(r'$\mathrm{Wavelength} \; \mathrm{(\mu m)}$', fontsize = 15)
        
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    
    legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':8}, ncol=2)
    
    plt.savefig('./H-_bound_free.pdf', bbox_inches='tight', fmt='pdf', dpi=1000)
    
    plt.show()
    
    plt.clf()
    
    #***** Now plot H- free-free opacity *****
    
    # Initialise plot
    ax = plt.gca()    
    #ax.set_xscale("log")
        
    xmajorLocator   = MultipleLocator(10.0)
    xmajorFormatter = FormatStrFormatter('%.1f')
    xminorLocator   = MultipleLocator(0.2)
    xminorFormatter = NullFormatter()
        
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_major_formatter(xmajorFormatter)
    ax.xaxis.set_minor_locator(xminorLocator)
    ax.xaxis.set_minor_formatter(xminorFormatter)
    
    # Plot cross section
    plt.semilogy(wl, alpha_ff, lw=1.0, alpha = 0.5, color= 'royalblue', label = r'H- free-free')
    
    plt.ylim([8.0e-50, 2.0e-47])
    plt.xlim([min(wl), max(wl)])
    plt.ylabel(r'$\mathrm{Binary \, \, Cross \, \, Section \, \, (m^{5} \, / n_{H} / n_{e-})}$', fontsize = 15)
    plt.xlabel(r'$\mathrm{Wavelength} \; \mathrm{(\mu m)}$', fontsize = 15)
        
    ax.set_xticks([0.4, 1.0, 2.0, 3.0, 4.0, 5.0])
    
    ax.text(min(wl_grid)*1.20, 1.5e-47, (r'$\mathrm{T = }$' + str(T_desired) + r'$\mathrm{K}$'), fontsize = 14)

    legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':8}, ncol=1)
    
    plt.savefig('./H-_free_free_' + str(T_desired) + 'K.pdf', bbox_inches='tight', fmt='pdf', dpi=1000)
    

#***** Begin main program ***** 
    
# Specify which molecules you want to extract from the database (full list available in the readme)
chemical_species = np.array(['H2O', 'CH4', 'NH3', 'HCN', 'CO', 'CO2'])
#chemical_species = np.array(['HCN'])

# Specify collisionally-induced absorption (CIA) pairs
cia_pairs = np.array(['H2-H2', 'H2-He'])

# Specify temperature and pressure grids on which to evaluate cross sections. Example:
P = np.logspace(-6, 2.0, 10)            # 20 pressures logarithmically spaced from 10^-6 -> 10^2 bar.
T = np.linspace(1000.0, 3000.0, 100)    # 100 temperatures linearly spaced from 800K -> 3000 K.

# Specify wavelength grid to extract cross section onto
wl_min = 0.4  # Minimum wavelength of grid (micron)
wl_max = 5.0  # Maximum wavelength of grid (micron)
N_wl = 1000   # Number of wavelength points

wl = np.linspace(wl_min, wl_max, N_wl)  # Uniform grid used here for demonstration purposes   

# Can use any P,T,wl grids, doesn't have to be uniform, but MUST be a numpy array.
# Note: if you use a v. large number of (P,T,wl) points, the memory usage could be quite high!
# This example has 10x100x1000 = 10^6 cross section values stored for each species.

# Either sample the nearest wavelength points from the high resolution (R~10^6) cross section database or use an averaging prescription 
opacity_treatment = 'Log-avg'           # Options: Opacity-sample / Log-avg
#opacity_treatment = 'Opacity-sample'   # Opacity sampling is faster, but for low-resolution wavelength grids log averaging is recommended
  
# Extract desired cross sections from the database
cross_sections, cia_absorption = Extract_opacity(chemical_species, cia_pairs, P, T, wl, opacity_treatment)   # Format: np array(N_species, N_P, N_T, N_wl) / Units: (m^2 / species)

# Evaluate H- bound-free and free-free opacities
alpha_bf = H_minus_bound_free(wl)
alpha_ff = H_minus_free_free(wl, T)

# Specify pressure and temperature where cross section plot is desired
P_desired = 1.0e-3
T_desired = 2000.0
P_idx = np.argmin(np.abs(P - P_desired))  # Closest index in the P array to desired value
T_idx = np.argmin(np.abs(T - T_desired))  # Closest index in the T array to desired value

# Example: seperate H2O cross section at 1 mbar and 1000K, and print to terminal
#H2O_cross_section = cross_sections[(np.where(chemical_species=='H2O')[0][0]),P_idx,T_idx,:]    # Format: np array(N_wl) / Units: (m^2 / molecule)
#print (H2O_cross_section)

# Plot cross sections at the desired P and T
plot_opacity(chemical_species, cia_pairs, cross_sections[:,P_idx,T_idx,:], 
             cia_absorption[:,T_idx,:], alpha_bf, alpha_ff[:,T_idx], 
             P_desired, T_desired, wl)

