## readinLatLong.py
## Functions for getting the latitde and longitude from a list of folders
## created by cloud academy participants.
## 2018.09.25 - c.j.baxter@uva.nl
## 2018.09.26 - modified by liac@umich.edu
##   to provide single function that does all the things
##
## Requirements:
##    numpy, glob, re
##------------------------------------------------------

import numpy as np
import glob
import re

# Function which cuts N characters of a string after a substrong
def cut_string(fullstring, substring, Nchar):
    newstring = fullstring[fullstring.index(substring):
                        fullstring.index(substring) + Nchar]
    return newstring

# Function to find the numbers in a string
def find_numbers(string, ints=True):
    numexp = re.compile(r'[-]?\d[\d,]*[\.]?[\d{2}]*') #optional - in front
    numbers = numexp.findall(string)
    numbers = [x.replace(',','') for x in numbers]
    if ints:
        res =  [int(x.replace(',','').split('.')[0]) for x in numbers][0]
    else:
        res =  numbers[0]
    return res

# Retrieves folder names for every available latitude an longitude
# (aka all the things)
def get_foldernames(root='Les_Houches_Cloud_Activity/HATP_7b'):
    """
    Reads in all the available folder names in order to find latitutde,
    longitude, and corresponding folder name.

    Parameters
    ----------
    root : String
        Root folder and filename prefix

    Returns
    -------
    A dictionary where the keys are tuples of strings (latitude, longitude),
    and the values are the strings with the associated folder name.
    """
    # All the folders containing drift files
    folders = glob.glob(root + '*')
    #print(folders)

    # Loop over all the folders and creatte a list of lattitudes and longitudes
    latlong = []
    for folder in folders:
        phi = find_numbers(cut_string(folder,'Phi', 7), ints=False)
        theta = find_numbers(cut_string(folder,'Theta', 9), ints=False)
        latlong.append((phi, theta))

    fnames = []
    for phi, theta in latlong:
        fnames.append(root + '_Phi{}Theta{}/'.format(float(phi),float(theta)))

    return dict(zip(latlong, fnames))
