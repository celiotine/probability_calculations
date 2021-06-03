import pandas as pd
import numpy as np
import scipy
import glob
import os
import sys
import re
import rate_functions as rates
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
import rate_functions as functions   ### Mike's code

#-----------------------------------------------------------------------------------#

N = int(sys.argv[1])  ## size of sampled population
population_path = sys.argv[2]  ## path to directory of COSMIC populations at different metallicities
save_path = sys.argv[3]   ## path to directory to save population

all_COSMIC_runs = os.listdir(population_path)

Zsun = 0.017
lowZ = Zsun/200 #lower bound on metallicity                              
highZ = Zsun #upper bound on metallicity                                                     
sigmaZ = 0.5 #sigma of the lognormal distribution about the mean metallicity                            

bpp_pops = []
initC_files = []
mets = []

## read-in all COSMIC populations                                                                              
for COSMIC_run in all_COSMIC_runs:
    ## read bpp data
    #data = glob.glob(COSMIC_run + "/*.hdf")
    bpp = pd.read_hdf(population_path + "/" + COSMIC_run + "/bpp_cut.hdf", key='bpp')
    bpp = bpp.drop(columns=['aj_1', 'aj_2', 'tms_1', 'tms_2', 'massc_1', 'massc_2', 'rad_1', 'rad_2', 'teff_1', 'teff_2', 'radc_1', 'radc_2', 'menv_1', 'menv_2', 'renv_1', 'renv_2', 'omega_spin_1', 'omega_spin_2', 'B_1', 'B_2', 'bacc_1', 'bacc_2', 'tacc_1', 'tacc_2', 'bhspin_1', 'bhspin_2'])
    bpp_pops.append(bpp)

    initC = pd.read_csv(population_path + "/" + COSMIC_run + "/initC.csv")
    initC_files.append(initC)

    met_val = float(re.findall('\d+.\d+', COSMIC_run)[0])
    met_format_str = "{:.8f}".format(met_val)
    mets.append(float(met_format_str))  ## run names are the population metallicity


print ("starting sampling...")

## grid of BBH formation redshifts                                      
zf_grid = np.linspace(0, 20, 10000)
SFR = rates.sfr_z(zf_grid, mdl='2017')
dVdz = cosmo.differential_comoving_volume(zf_grid)
zf_weights = SFR * dVdz * (1+zf_grid)**-1

zf_picks = np.random.choice(zf_grid, N, replace=True, p=zf_weights/np.sum(zf_weights))

## sample a metallicity for each sampled formation redshift 
df = pd.DataFrame()
initC_df = pd.DataFrame()

for zf_pick in zf_picks:
    
    ### get metallicity weights for all drawn formation redshifts                                              
    Z_weights = functions.metal_disp_z(zf_pick, np.array(mets), sigmaZ, lowZ, highZ)

    ## draw metallicity using the above weights to choose a COSMIC population to draw a system from             
    Z_pick = np.random.choice(mets, p=Z_weights/np.sum(Z_weights))

    pop_idx = np.where(mets==Z_pick)[0][0]
    this_pop = bpp_pops[pop_idx]   ## the population of DCOs from which to draw a system; this is the full bpp file from that COSMIC run
    this_initC = initC_files[pop_idx]

    ids = np.unique(this_pop['bin_num'].values)
    this_binary = np.random.choice(ids, replace=True)  ## the sampled binary from all COSMIC populations
    
    this_pop.loc[this_binary, "metallicity"] = Z_pick
    this_pop.loc[this_binary, "redshift"] = zf_pick
    
    df = df.append(this_pop.loc[this_binary])
    initC_df = initC_df.append(this_initC.loc[this_initC['bin_num'] == this_binary])

df.to_csv(save_path + "sampled_large_population.csv", index=False)
initC_df.to_csv(save_path + "sampled_large_initC.csv", index=False)    
