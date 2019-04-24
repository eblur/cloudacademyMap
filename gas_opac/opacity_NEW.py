# Demostrates how to open the opacity database and interpolate to a given P, T, and wavelength grid
# Author: Ryan J. MacDonald (w/ improvements from Lia Corrales)

# V1.0: Evaluates cross sections at a single P,T point (8th November 2018)
# V2.0: Evaluates cross sections on a grid of (P,T) points (6th December 2018)
# V3.0: Evaluates collisionally-induced opacity on a grid of (T) points (17th February 2019)
# V4.0: Inclusion of H- bound-free and free-free opacities (26th February 2019)
# V5.0: Optimised for speed wby interpolating directly to layer conditions (5th April 2019)

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

@jit(nopython=True)
def interpolate_cia(log_cia, nu_l, nu_out, nu_r, nu_cia, T, T_grid_cia, N_T_cia, N_D, N_wl_out, N_nu_out, y, w_T, mode):
    
    ''' Interpolates a collisionally-induced absorption (CIA) binary cross 
        section onto the T value in each layer of the model atmosphere.
        
        The input is in wavenumnber to take advantage of fast prior index 
        location on a uniform grid, which wouldn't work for the (non-uniform) 
        wavelength grid. Array reversal to output in increasing wavelength is 
        handled by indexing by a factor of (N_wl-1)-k throughout.
        
        Wavelength initialisation is handled via either opacity sampling
        (choosing nearest pre-computed wavelength point) or via averaging
        the logarithm of the cross section over the wavelength bin range
        surrounding each wavelength on the output model wavelength grid.
        
    '''
   
    cia_inp = np.zeros(shape=(N_D, N_wl_out))   # Initialise output CIA cross section
    
    N_nu_cia = len(nu_cia)   # Number of wavenumber points in CIA array

    for i in xrange(N_D):    # Loop over layer temperatures
    
        T_i = T[i]               # Temperature to interpolate to
        T1 = T_grid_cia[y[i]]    # Closest lower temperaure on CIA opacity grid 
        T2 = T_grid_cia[y[i]+1]  # Closest higher temperaure on CIA opacity grid 
        
        for k in xrange(N_nu_out):  # Loop over wavenumbers
            
            # Find closest index on CIA wavenumber grid to each model wavenumber
            z_l = closest_index(nu_l[k], nu_cia[0], nu_cia[-1], N_nu_cia)
            z = closest_index(nu_out[k], nu_cia[0], nu_cia[-1], N_nu_cia)
            z_r = closest_index(nu_r[k], nu_cia[0], nu_cia[-1], N_nu_cia)
            
            # If this wavenumber falls out of range of opacity grid, set opacity to zero
            if ((z_l == 0) or (z_r == (N_nu_cia-1))):
                cia_inp[i, ((N_wl_out-1)-k)] = 0.0
                
            else:  # Otherwise, proceed with interpolation
                
                # Opacity sampling
                if (mode == 1):  
                    
                    # If layer T below min value on CIA grid (200 K), set CIA to value at min T on opac grid
                    if (y[i] == -1):
                        cia_inp[i, ((N_wl_out-1)-k)] = 10 ** (log_cia[0, z])
                            
                    # If layer T above max value (3500 K), set CIA to value at max T on opac grid
                    elif (y[i] == -2):
                        cia_inp[i, ((N_wl_out-1)-k)] = 10 ** (log_cia[(N_T_cia-1), z])
                        
                    # Interpolate CIA to temperature in this layer
                    else: 
                        cia_reduced = 10 ** (log_cia[y[i]:y[i]+2, z])
                        cia_T1, cia_T2 = cia_reduced[0], cia_reduced[1]    # CIA(T1)[j,z], CIA(T2)[j,z]
                                
                        cia_inp[i, ((N_wl_out-1)-k)] = (np.power(cia_T1, (w_T[i]*((1.0/T2) - (1.0/T_i)))) *
                                                        np.power(cia_T2, (w_T[i]*((1.0/T_i) - (1.0/T1)))))
                
                # Log averaging
                elif (mode == 2):
                    
                    # If layer T below min value on CIA grid (200 K), set CIA to value at min T on opac grid
                    if (y[i] == -1):
                        cia_in_bin = np.mean(log_cia[0, z_l:z_r+1])
                        cia_inp[i, ((N_wl_out-1)-k)] = 10 ** (cia_in_bin)
                            
                    # If layer T above max value (3500 K), set CIA to value at max T on opac grid
                    elif (y[i] == -2):
                        cia_in_bin = np.mean(log_cia[(N_T_cia-1), z_l:z_r+1])
                        cia_inp[i, ((N_wl_out-1)-k)] = 10 ** (cia_in_bin)
                        
                    # Interpolate CIA to temperature in this layer
                    else: 
                        cia_in_bin_T1 = 10 ** np.mean(log_cia[y[i], z_l:z_r+1])        # CIA in bin at temperaure T1
                        cia_in_bin_T2 = 10 ** np.mean(log_cia[y[i]+1, z_l:z_r+1])      # CIA in bin at temperaure T2
                                
                        cia_inp[i, ((N_wl_out-1)-k)] = (np.power(cia_in_bin_T1, (w_T[i]*((1.0/T2) - (1.0/T_i)))) *
                                                        np.power(cia_in_bin_T2, (w_T[i]*((1.0/T_i) - (1.0/T1)))))
                                
    return cia_inp

@jit(nopython=True)
def interpolate_sigma(log_sigma, nu_l, nu_out, nu_r, nu_opac, P, T, N_D, log_P_grid, T_grid, N_P_opac, N_T_opac, N_wl_out, N_nu_out, y, w_T, mode):
    
    ''' Interpolates a cross section onto the (P,T) values in each layer of
        the model atmosphere.
        
       The input is in wavenumnber to take advantage of fast prior index location 
       on a uniform grid, which wouldn't work for the (non-uniform) wavelength grid
       Array reversal to output in increasing wavelength is handled by indexing
       by a factor of (N_wl-1)-k throughout. 
        
    '''
   
    sigma_inp = np.zeros(shape=(N_D, N_wl_out))   # Initialise output cross section
    
    N_nu_opac = len(nu_opac)   # Number of wavenumber points in opacity array
    
    #***** Firstly, find pressure interpolation weighting factors *****#
    log_P = np.log10(P)  # Log of model pressure grid
    
    # Array of indicies on opacity pressure opacity grid prior to model atmosphere layer pressures
    x = np.zeros(N_D).astype(np.int64) 
    
    w_P = np.zeros(N_D)  # Pressure weights
    
    # Useful functions of weights for interpolation
    b1 = np.zeros(shape=(N_D))
    b2 = np.zeros(shape=(N_D))
            
    # Find closest P indicies in opacity grid corresponding to model layer pressures
    for i in xrange(N_D):
        
        # If pressure below minimum, do not interpolate
        if (log_P[i] < log_P_grid[0]):
            x[i] = -1      # Special value (1) used in opacity initialiser
            w_P[i] = 0.0
        
        # If pressure above maximum, do not interpolate
        elif (log_P[i] >= log_P_grid[-1]):
            x[i] = -2      # Special value (2) used in opacity initialiser
            w_P[i] = 0.0
        
        else:
            x[i] = prior_index_V2(log_P[i], log_P_grid[0], log_P_grid[-1], N_P_opac)
        
            # Weights - fractional distance along pressure axis of sigma array
            w_P[i] = (log_P[i]-log_P_grid[x[i]])/(log_P_grid[x[i]+1]-log_P_grid[x[i]])     
         
        # Precalculate interpolation pre-factors to reduce computation overhead
        b1[i] = (1.0-w_P[i])
        b2[i] = w_P[i] 
        
    # Note: temperaure interpolation indicies and weights passed through function arguments

    # Begin interpolation procedure
    for i in xrange(N_D):   # Loop over model layers
        
        T_i = T[i]           # Layer temperature to interpolate to
        T1 = T_grid[y[i]]    # Closest lower temperaure on opacity grid 
        T2 = T_grid[y[i]+1]  # Closest higher temperaure on opacity grid 
        
        for k in xrange(N_nu_out):   # Loop over model wavenumbers
            
            # Calculate closest indicies to output wavenumber bin left, centre, and right edge
            z_l = closest_index(nu_l[k], nu_opac[0], nu_opac[-1], N_nu_opac)
            z = closest_index(nu_out[k], nu_opac[0], nu_opac[-1], N_nu_opac)
            z_r = closest_index(nu_r[k], nu_opac[0], nu_opac[-1], N_nu_opac)
            
            # If this wavenumber falls out of range of opacity grid, set opacity to zero
            if ((z_l == 0) or (z_r == (N_nu_opac-1))):
                sigma_inp[i, ((N_wl_out-1)-k)] = 0.0
                
            else:  # Otherwise, proceed with interpolation
                
                # Opacity sampling
                if (mode == 1):  

                    # Find rectangle of stored opacity points located at [log_P1, log_P2, T1, T2, z]
                    log_sigma_PT_rectangle = log_sigma[x[i]:x[i]+2, y[i]:y[i]+2, z]
    
                    # Pressure interpolation is handled first, followed by temperaure interpolation
                    # First, check for off-grid special cases
                    
                    # If layer P below minimum on opacity grid (1.0e-6 bar), set value to edge case
                    if (x[i] == -1):      
                        
                        # If layer T also below minimum on opacity grid (100 K), set value to edge case
                        if (y[i] == -1):
                            sigma_inp[i, ((N_wl_out-1)-k)] = 10 ** (log_sigma[0, 0, z])  # No interpolation needed
                         
                        # If layer T above maximum on opacity grid (3500 K), set value to edge case
                        elif (y[i] == -2):
                            sigma_inp[i, ((N_wl_out-1)-k)] = 10 ** (log_sigma[0, (N_T_opac-1), z])  # No interpolation needed
                            
                        # If desired temperaure is on opacity grid, set T1 and T2 values to those at min P on grid
                        else:
                            sig_T1 = 10 ** (log_sigma[0, y[i], z])     
                            sig_T2 = 10 ** (log_sigma[0, y[i]+1, z])
                            
                            # Only need to interpolate over temperaure interpolate cross section to layer temperature                    
                            sigma_inp[i, ((N_wl_out-1)-k)] =  (np.power(sig_T1, (w_T[i]*((1.0/T2) - (1.0/T_i)))) *
                                                               np.power(sig_T2, (w_T[i]*((1.0/T_i) - (1.0/T1)))))
                        
                    # If layer P above maximum on opacity grid (100 bar), set value to edge case
                    elif (x[i] == -2):
                        
                        # If layer T below minimum on opacity grid (100 K), set value to edge case
                        if (y[i] == -1):
                            sigma_inp[i, ((N_wl_out-1)-k)] = 10 ** (log_sigma[(N_P_opac-1), 0, z])  # No interpolation needed
                         
                        # If layer T also above maximum on opacity grid (3500 K), set value to edge case
                        elif (y[i] == -2):
                            sigma_inp[i, ((N_wl_out-1)-k)] = 10 ** (log_sigma[(N_P_opac-1), (N_T_opac-1), z])  # No interpolation needed
                            
                        # If desired temperaure is on opacity grid, set T1 and T2 values to those at maximum P on grid             
                        else:
                            sig_T1 = 10 ** (log_sigma[(N_P_opac-1), y[i], z])
                            sig_T2 = 10 ** (log_sigma[(N_P_opac-1), y[i]+1, z])
                            
                            # Now interpolate cross section to layer temperature                    
                            sigma_inp[i, ((N_wl_out-1)-k)] =  (np.power(sig_T1, (w_T[i]*((1.0/T2) - (1.0/T_i)))) *
                                                               np.power(sig_T2, (w_T[i]*((1.0/T_i) - (1.0/T1)))))
                
                    # If both desired P and T are on opacity grid (should be true in most cases!)
                    else:
                        
                        # Interpolate log(cross section) in log(P), then power to get interpolated values at T1 and T2
                        sig_T1 =  10 ** (b1[i]*(log_sigma_PT_rectangle[0,0]) +       # Cross section at T1
                                         b2[i]*(log_sigma_PT_rectangle[1,0]))
                        sig_T2 =  10 ** (b1[i]*(log_sigma_PT_rectangle[0,1]) +       # Cross section at T2
                                         b2[i]*(log_sigma_PT_rectangle[1,1]))
            
                        # Now interpolate cross section to layer temperature                    
                        sigma_inp[i, ((N_wl_out-1)-k)] =  (np.power(sig_T1, (w_T[i]*((1.0/T2) - (1.0/T_i)))) *
                                                           np.power(sig_T2, (w_T[i]*((1.0/T_i) - (1.0/T1)))))
                        
                # Log averaging
                elif (mode == 2):
                    
                    # If layer P below minimum on opacity grid (1.0e-6 bar), set value to edge case
                    if (x[i] == -1):      
                        
                        # If layer T also below minimum on opacity grid (100 K), set value to edge case
                        if (y[i] == -1):
                            sigma_in_bin = np.mean(log_sigma[0, 0, z_l:z_r+1])
                            sigma_inp[i, ((N_wl_out-1)-k)] = 10 ** (sigma_in_bin)  # No interpolation needed
                         
                        # If layer T above maximum on opacity grid (3500 K), set value to edge case
                        elif (y[i] == -2):
                            sigma_in_bin = np.mean(log_sigma[0, (N_T_opac-1), z_l:z_r+1])
                            sigma_inp[i, ((N_wl_out-1)-k)] = 10 ** (sigma_in_bin)  # No interpolation needed
                            
                        # If desired temperaure is on opacity grid, set T1 and T2 values to those at min P on grid
                        else:
                            sig_T1 = 10 ** (np.mean(log_sigma[0, y[i], z_l:z_r+1]))  
                            sig_T2 = 10 ** (np.mean(log_sigma[0, y[i]+1, z_l:z_r+1]))
                            
                            # Only need to interpolate over temperaure interpolate cross section to layer temperature                    
                            sigma_inp[i, ((N_wl_out-1)-k)] =  (np.power(sig_T1, (w_T[i]*((1.0/T2) - (1.0/T_i)))) *
                                                               np.power(sig_T2, (w_T[i]*((1.0/T_i) - (1.0/T1)))))
                            
                    # If layer P above maximum on opacity grid (100 bar), set value to edge case
                    elif (x[i] == -2):
                        
                        # If layer T below minimum on opacity grid (100 K), set value to edge case
                        if (y[i] == -1):
                            sigma_in_bin = np.mean(log_sigma[(N_P_opac-1), 0, z_l:z_r+1])
                            sigma_inp[i, ((N_wl_out-1)-k)] = 10 ** (sigma_in_bin)  # No interpolation needed
                         
                        # If layer T also above maximum on opacity grid (3500 K), set value to edge case
                        elif (y[i] == -2):
                            sigma_in_bin = np.mean(log_sigma[(N_P_opac-1), (N_T_opac-1), z_l:z_r+1])
                            sigma_inp[i, ((N_wl_out-1)-k)] = 10 ** (sigma_in_bin)  # No interpolation needed
                            
                        # If desired temperaure is on opacity grid, set T1 and T2 values to those at maximum P on grid             
                        else:
                            sig_T1 = 10 ** (np.mean(log_sigma[(N_P_opac-1), y[i], z_l:z_r+1]))  
                            sig_T2 = 10 ** (np.mean(log_sigma[(N_P_opac-1), y[i]+1, z_l:z_r+1]))
                            
                            # Now interpolate cross section to layer temperature                    
                            sigma_inp[i, ((N_wl_out-1)-k)] =  (np.power(sig_T1, (w_T[i]*((1.0/T2) - (1.0/T_i)))) *
                                                               np.power(sig_T2, (w_T[i]*((1.0/T_i) - (1.0/T1)))))
                            
                    # If both desired P and T are on opacity grid (should be true in most cases!)
                    else:
                        
                        sig_in_bin_P1_T1 = np.mean(log_sigma[x[i], y[i], z_l:z_r+1])      # Cross section in bin at P1 and T1
                        sig_in_bin_P2_T1 = np.mean(log_sigma[x[i]+1, y[i], z_l:z_r+1])    # Cross section in bin at P2 and T1
                        sig_in_bin_P1_T2 = np.mean(log_sigma[x[i], y[i]+1, z_l:z_r+1])    # Cross section in bin at P1 and T2
                        sig_in_bin_P2_T2 = np.mean(log_sigma[x[i]+1, y[i]+1, z_l:z_r+1])  # Cross section in bin at P2 and T2
                        
                        # Interpolate log(cross section) in log(P), then power to get interpolated values at T1 and T2
                        sig_T1 =  10 ** (b1[i]*(sig_in_bin_P1_T1) +       # Cross section at T1
                                         b2[i]*(sig_in_bin_P2_T1))
                        sig_T2 =  10 ** (b1[i]*(sig_in_bin_P1_T2) +       # Cross section at T2
                                         b2[i]*(sig_in_bin_P2_T2))
            
                        # Now interpolate cross section to layer temperature                    
                        sigma_inp[i, ((N_wl_out-1)-k)] =  (np.power(sig_T1, (w_T[i]*((1.0/T2) - (1.0/T_i)))) *
                                                           np.power(sig_T2, (w_T[i]*((1.0/T_i) - (1.0/T1)))))

    return sigma_inp

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
def H_minus_free_free(T_arr, wl_um):
    
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
    alpha_ff = np.zeros(shape=(len(T_arr), len(wl_um)))
        
    for i, T in enumerate(T_arr):
            
        theta = 5040.0/T      # Reciprocal temperature commonly used in these fits
        kT = 1.38066e-16 * T  # Boltzmann constant * temperature (erg)
    
        for k, wl in enumerate(wl_um):
                
            # Range of validity of fit (wl > 0.182 um)
            if (wl < 0.182):
                    
                alpha_ff[i,k] = 1.0e-250   # Small value (proxy for zero, but avoids log(0) issues)
            
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
                    alpha_ff[i,k] += 1.0e-29 * (np.power(theta, ((n + 1.0)/2.0)) * 
                                                (A[n-1]*(wl**2) + B[n-1] + C[n-1]/wl + 
                                                 D[n-1]/(wl**2) + E[n-1]/(wl**3) + 
                                                 F[n-1]/(wl**4))) * kT  # cm^5 / H / e-
                    
                alpha_ff[i,k] *= 1.0e-10  # Convert cm^5 to m^5
                    
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
                    alpha_ff[i,k] += 1.0e-29 * (np.power(theta, ((n + 1.0)/2.0)) * 
                                                (A[n-1]*(wl**2) + B[n-1] + C[n-1]/wl + 
                                                 D[n-1]/(wl**2) + E[n-1]/(wl**3) + 
                                                 F[n-1]/(wl**4))) * kT  # cm^5 / H / e-
                alpha_ff[i,k] *= 1.0e-10  # Convert cm^5 to m^5
    
    return alpha_ff

def Extract_opacity_NEW(chemical_species, cia_pairs, P, T, wl_out, opacity_treatment, root='../'):
    
    ''' Read in all opacities and interpolate onto the pressure and 
        temperature in each layer for desired wavelength grid
    
    '''
    
    print("Now reading in cross sections")
    
    # First, check from config.py which opacity calculation mode is specified
    if   (opacity_treatment == 'Opacity-sample'): calculation_mode = 1
    elif (opacity_treatment == 'Log-avg'):        calculation_mode = 2
    
    #***** Firstly, we initialise the various quantities needed *****#
    
    # Open HDF5 files containing molecular + atomic opacities and CIA
    opac_file = h5py.File(root + 'Opacity_database_0.01cm-1.hdf5', 'r')
    cia_file = h5py.File(root + 'Opacity_database_cia.hdf5', 'r')
    
    # Convert model wavelength grid to wavenumber grid
    nu_out = 1.0e4/wl_out      # Model wavenumber grid (cm^-1)
    nu_out = nu_out[::-1]      # Reverse direction to increase with wavenumber
    
    N_nu_out = len(nu_out)    # Number of wavenumbers on output grid
    N_wl_out = len(wl_out)    # Number of wavelengths on output grid
    
    N_D = len(P)  # No. of depth points (layers) in atmosphere
    
    # Find wavenumbers at left and right edge of each bin on outpur grid
    # Vectorized by Lia
    nu_edges = np.append(nu_out[0] - (nu_out[1] - nu_out[0]), nu_out)
    nu_edges = np.append(nu_edges, nu_out[-1] + (nu_out[-1] - nu_out[-2]))
    nu_l = 0.5 * (nu_edges[:-2] + nu_edges[1:-1])
    nu_r = 0.5 * (nu_edges[1:-1] +  nu_edges[2:])

    #***** Process collisionally Induced Absorption (CIA) *****#
        
    # Initialise cia opacity array, interpolated to temperature in each model layer
    cia_stored = dict()
        
    for q in cia_pairs:
            
        #***** Read in T grid used in this CIA file*****#
        T_grid_cia_q = np.array(cia_file[q + '/T'])
        N_T_cia_q = len(T_grid_cia_q)  # Number of temperatures in this grid
            
        # Read in wavenumber array used in this CIA file
        nu_cia_q = np.array(cia_file[q + '/nu'])
        
        # Find T index in CIA arrays prior to each layer temperature
        y_cia_q = np.zeros(N_D, dtype=np.int)
        w_T_cia_q = T_interpolation_init(N_D, T_grid_cia_q, T, y_cia_q)
            
        # Read in log10(binary cross section) for specified CIA pair
        log_cia_q = np.array(cia_file[q + '/log(cia)']).astype(np.float64)   
            
        # Interpolate CIA to temperature in each atmospheric layer
        cia_stored[q] = interpolate_cia(log_cia_q, nu_l, nu_out, nu_r, nu_cia_q, T, T_grid_cia_q, N_T_cia_q, N_D, N_wl_out, N_nu_out, y_cia_q, w_T_cia_q, calculation_mode)
            
        del log_cia_q, nu_cia_q, w_T_cia_q, y_cia_q  # Clear raw cross section to free up memory
            
        print(q + " done")
            
    cia_file.close()
            
    #***** Process molecular and atomic opacities *****#
    
    # Initialise molecular and atomic opacity dictionary, interpolated to temperatures and pressures in each model layer
    sigma_stored = dict()
        
    # Load molecular and atomic absorption cross sections
    for q in chemical_species:
            
        #***** Read in grids used in this opacity file *****#
        T_grid_q = np.array(opac_file[q + '/T'])   
        log_P_grid_q = np.array(opac_file[q + '/log(P)'])   # Units: log10(P/bar)!
            
        # Read in wavenumber array used in this opacity file
        nu_q = np.array(opac_file[q + '/nu'])
            
        N_T_q = len(T_grid_q)      # Number of temperatures in this grid
        N_P_q = len(log_P_grid_q)  # Number of pressures in this grid
            
        # Find T index in cross secion arrays prior to each layer temperature
        y_q = np.zeros(N_D, dtype=np.int)
        w_T_q = T_interpolation_init(N_D, T_grid_q, T, y_q)   
            
        # Read in log10(cross section) of specified molecule
        log_sigma_q = np.array(opac_file[q + '/log(sigma)']).astype(np.float64)      
                
        # Interpolate cross section to (P,T) in each atmospheric layer
        sigma_stored[q] = interpolate_sigma(log_sigma_q, nu_l, nu_out, nu_r, nu_q, P, T, N_D, log_P_grid_q, T_grid_q, N_P_q, N_T_q, N_wl_out, N_nu_out, y_q, w_T_q, calculation_mode)

        del log_sigma_q, nu_q, w_T_q, y_q   # Clear raw cross section to free up memory
            
        print(q + " done")
            
    opac_file.close()
    
    #***** Compute H- bound-free and free-free opacities *****#
    
    alpha_bf = H_minus_bound_free(wl_out)
    alpha_ff = H_minus_free_free(T, wl_out)
    
    print("H- done")
            
    return sigma_stored, cia_stored, alpha_bf, alpha_ff


#***** Functions below not currently used, preserved for reference ******
    
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
    
## Written by Lia
## To separate out reading step
def load_db(filename='Opacity_database_0.01cm-1.hdf5', root='../'):

    open_filename = root + filename
    print("Reading opacity database file")
    opac_file = h5py.File(open_filename, 'r')

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

def Extract_opacity_Lia(chemical_species, P, T, wl_out, opacity_treatment):
    
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
def Extract_opacity_PTpairs(chemical_species, P, T, wl_out, opacity_treatment, root='../'):
    
    '''Convienient function to read in all opacities and pre-interpolate
       them onto the desired pressure, temperature, and wavelength grid'''

    assert len(P) == len(T)

    T_grid, log_P_grid, nu_opac, opac_file = load_db(root=root)

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

# ***** Functions from here on used for diagnostic plotting of cross sections *****#

def plot_opacity(chemical_species, cia_pairs, sigma_stored, cia_stored, alpha_bf, alpha_ff,
                 P, T, wl_grid, savefigs=False, **kwargs):
    
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
        sigma_plt = sigma_stored[species][0,:]   # Cross section of this species at given (P,T) pair (m^2)
        
        if (smooth == True):
            sigma_plt = gauss_conv(sigma_plt, sigma=smooth_factor, mode='nearest')
            
        # Plot cross section
        plt.semilogy(wl_grid, sigma_plt, label=species, **kwargs)
    
    plt.ylim([1.0e-31, 1.0e-21])
    plt.xlim([min(wl_grid), max(wl_grid)])
    plt.ylabel(r'$\mathrm{Cross \, \, Section \, \, (m^{2} \, mol^{-1})}$', fontsize = 15)
    plt.xlabel(r'$\mathrm{Wavelength} \; \mathrm{(\mu m)}$', fontsize = 15)
    
    ax.text(min(wl_grid)*1.20, 2.0e-22, (r'$\mathrm{T = }$' + str(T) + \
                                         r'$\mathrm{K \, \, P = }$' + \
                                         str(P*1000) + r'$\mathrm{mbar}$'), fontsize = 14)
    
    legend = plt.legend(loc='upper right', frameon=False, prop={'size':6}, ncol=2)
    
    '''for legline in legend.legendHandles:
    legline.set_linewidth(1.0)'''

    if savefigs:
        plt.savefig('./cross_sections_' + str(T) + 'K_' + str(P*1000) + 'mbar_TEST.pdf', bbox_inches='tight', fmt='pdf', dpi=1000)

    plt.show()
    
    plt.clf()
    
    #***** Now plot CIA *****
    
    #colours_cia = np.array(['royalblue', 'crimson'])
    
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
    for pair in cia_pairs:

        cia_plt = cia_stored[pair][0,:]  # Cross section of species q at given (P,T) pair (m^5)
        
        # Plot cross section
        plt.semilogy(wl_grid, cia_plt, label=pair, **kwargs)
    
    plt.ylim([1.0e-60, 1.0e-52])
    plt.xlim([min(wl_grid), max(wl_grid)])
    plt.ylabel(r'$\mathrm{Binary \, \, Cross \, \, Section \, (CIA) \, \, (m^{5} mol^{-2})}$', fontsize = 15)
    plt.xlabel(r'$\mathrm{Wavelength} \; \mathrm{(\mu m)}$', fontsize = 15)
    
    ax.text(min(wl_grid)*1.20, 3.0e-53, (r'$\mathrm{T = }$' + str(T) + r'$\mathrm{K}$'), fontsize = 14)
    
    legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':8})
    
    '''for legline in legend.legendHandles:
        legline.set_linewidth(1.0)'''

    if savefigs:
        plt.savefig('./CIA_cross_sections_' + str(T) + 'K.pdf', bbox_inches='tight', fmt='pdf', dpi=1000)
    
    plt.show()
    
    plt.clf()
    
    #***** Now plot H- bound-free opacity *****
        
    # Initialise plot
    #ax = plt.gca()    
        
    ax = plt.subplot(111)
    #xmajorLocator   = MultipleLocator(0.5)
    #xmajorFormatter = FormatStrFormatter('%.1f')
    #xminorLocator   = MultipleLocator(0.1)
    #xminorFormatter = NullFormatter()
        
    #ax.xaxis.set_major_locator(xmajorLocator)
    #ax.xaxis.set_major_formatter(xmajorFormatter)
    #ax.xaxis.set_minor_locator(xminorLocator)
    #ax.xaxis.set_minor_formatter(xminorFormatter)
    
    # Plot cross section
    plt.semilogy(wl, alpha_bf, label = r'H- bound-free', **kwargs)
    
    plt.ylim([1.0e-23, 1.0e-20])
    plt.xlim([min(wl), 1.8])
    plt.ylabel(r'$\mathrm{Cross \, \, Section \, \, (m^{2} \, / n_{H-})}$', fontsize = 15)
    plt.xlabel(r'$\mathrm{Wavelength} \; \mathrm{(\mu m)}$', fontsize = 15)
        
    #ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8])
    
    legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':8}, ncol=2)
    
    if savefigs:
        plt.savefig('./H-_bound_free.pdf', bbox_inches='tight', fmt='pdf', dpi=1000)
    
    plt.show()
    
    plt.clf()
    
    #***** Now plot H- free-free opacity *****
    
    # Initialise plot
    #ax = plt.gca()    
    #ax.set_xscale("log")
    
    ax = plt.subplot(111)
    #xmajorLocator   = MultipleLocator(10.0)
    #xmajorFormatter = FormatStrFormatter('%.1f')
    #xminorLocator   = MultipleLocator(0.2)
    #xminorFormatter = NullFormatter()
        
    #ax.xaxis.set_major_locator(xmajorLocator)
    #ax.xaxis.set_major_formatter(xmajorFormatter)
    #ax.xaxis.set_minor_locator(xminorLocator)
    #ax.xaxis.set_minor_formatter(xminorFormatter)
    
    # Plot cross section
    plt.semilogy(wl, alpha_ff, label = r'H- free-free', **kwargs)
    
    plt.ylim([8.0e-50, 2.0e-47])
    plt.xlim([min(wl), max(wl)])
    plt.ylabel(r'$\mathrm{Binary \, \, Cross \, \, Section \, \, (m^{5} \, / n_{H} / n_{e-})}$', fontsize = 15)
    plt.xlabel(r'$\mathrm{Wavelength} \; \mathrm{(\mu m)}$', fontsize = 15)
        
    #ax.set_xticks([0.4, 1.0, 2.0, 3.0, 4.0, 5.0])
    
    ax.text(min(wl_grid)*1.20, 1.5e-47, (r'$\mathrm{T = }$' + str(T) + r'$\mathrm{K}$'), fontsize = 14)

    legend = plt.legend(loc='upper right', shadow=False, frameon=False, prop={'size':8}, ncol=1)
    
    if savefigs:
        plt.savefig('./H-_free_free_' + str(T) + 'K.pdf', bbox_inches='tight', fmt='pdf', dpi=1000)
        
    plt.show()
    
    plt.clf()
    
    return

#***** Begin main program ***** 

# This code will execute if you run the script from the terminal
# Otherwise, no
if __name__ == '__main__':
    # Specify which molecules you want to extract from the database (full list available in the readme)
    chemical_species = np.array(['H2O', 'CH4', 'NH3', 'HCN', 'CO', 'CO2'])
    
    # Specify collisionally-induced absorption (CIA) pairs
    cia_pairs = np.array(['H2-H2', 'H2-He'])
    
    # Specify layer pressures and temperatures in each layer
    # For this plotting test, take same P and T throughout atmosphere
    P = np.ones(100) * 1.0e-3             # Pressures in each layer (bar)
    #P = np.logspace(-6.0, 2.0, 100.0)    # Pressures in each layer (bar)
    T = np.ones(100) * 2000.0             # Temperature in each layer (K)  

    # Specify wavelength grid to extract cross section onto
    wl_min = 0.4  # Minimum wavelength of grid (micron)
    wl_max = 5.0  # Maximum wavelength of grid (micron)
    N_wl = 1000   # Number of wavelength points

    wl = np.linspace(wl_min, wl_max, N_wl)  # Uniform grid used here for demonstration purposes   
    
    # Either sample the nearest wavelength points from the high resolution (R~10^6) cross section database or use an averaging prescription 
    opacity_treatment = 'Log-avg'           # Options: Opacity-sample / Log-avg
    #opacity_treatment = 'Opacity-sample'   # Opacity sampling is faster, but for low-resolution wavelength grids log averaging is recommended
    
    # Extract desired cross sections from the database
    cross_sections, cia_absorption, alpha_bf, alpha_ff = Extract_opacity_NEW(chemical_species, cia_pairs, P, T, wl, opacity_treatment)   # Format: np array(N_species, N_wl) / Units: (m^2 / species)

    # Evaluate H- bound-free and free-free opacities
    #alpha_bf = H_minus_bound_free(wl)
    #alpha_ff = H_minus_free_free(T, wl)
    
    # Example: seperate H2O cross section, and print to terminal
    #H2O_cross_section = cross_sections['H2O']    # Format: np array(N_wl) / Units: (m^2 / molecule)
    #print (H2O_cross_section)
    
    # Plot cross sections
    plot_opacity(chemical_species, cia_pairs, cross_sections, cia_absorption, 
                 alpha_bf, alpha_ff[0,:], P[0], T[0], wl, savefigs=True, alpha=0.8, lw=0.5)

