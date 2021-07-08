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
mets = [float(x) for x in all_COSMIC_runs]

Zsun = 0.017
lowZ = Zsun/200 #lower bound on metallicity                              
highZ = Zsun #upper bound on metallicity                                                     
sigmaZ = 0.5 #sigma of the lognormal distribution about the mean metallicity                            

## grid of BBH formation redshifts                                      
zf_grid = np.linspace(0, 20, 10000)
SFR = rates.sfr_z(zf_grid, mdl='2017')
dVdz = cosmo.differential_comoving_volume(zf_grid)
zf_weights = SFR * dVdz * (1+z_f)**-1

zf_picks = np.random.choice(zf_grid, N, replace=True, p=zf_weights/np.sum(zf_weights))

## sample a metallicity for each sampled formation redshift 
Z_picks = []
for zf_pick in zf_picks:
    
    ### get metallicity weights for all drawn formation redshifts                                              
    Z_weights = functions.metal_disp_z(zf_pick, np.array(mets), sigmaZ, lowZ, highZ)

    ## draw metallicity using the above weights to choose a COSMIC population to draw a system from             
    Z_pick = np.random.choice(mets, p=Z_weights/np.sum(Z_weights))
    Z_picks.append(Z_pick.item())

Z_picks = np.array(Z_picks)


## loop through COSMIC runs and sample systems
df = pd.DataFrame()
initC_df = pd.DataFrame()

for COSMIC_run in all_COSMIC_runs:
    
    met_val = float(COSMIC_run)
    print ("sampling ", met_val, " population")

    bpp = pd.read_hdf(population_path + "/" + COSMIC_run + "/bpp_cut.hdf", key='bpp')
    bpp = bpp.drop(columns=['aj_1', 'aj_2', 'tms_1', 'tms_2', 'massc_1', 'massc_2', 'rad_1', 'rad_2', 'teff_1', 'teff_2', 'radc_1',\
 'radc_2', 'menv_1', 'menv_2', 'renv_1', 'renv_2', 'omega_spin_1', 'omega_spin_2', 'B_1', 'B_2', 'bacc_1', 'bacc_2', 'tacc_1', 'tacc_2', 'bhspin_1', 'bhspin_2']) 

    initC = pd.read_csv(population_path + "/" + COSMIC_run + "/initC.csv") 

    bin_vals = np.unique(bpp['bin_num'])
    met_val = float(COSMIC_run)
    
    ## sample number of  binaries that have this population metallicity
    Z_locs = np.where(np.isclose(Z_picks, met_val))[0]

    num_at_Z = len(Z_locs)
    Z_redshifts = zf_picks[Z_locs]

    sampled_bin_vals = np.random.choice(bin_vals, num_at_Z, replace=True)
    sampled_bpp_systems = bpp.loc[sampled_bin_vals]

    sampled_initC_systems = initC.loc[initC['bin_num'].isin(sampled_bin_vals)]

    sampled_bpp_systems['metallicity'] = met_val
    sampled_bpp_systems['redshift'] = Z_redshifts[0]
    
    i = 0
    for bin_num in sampled_bin_vals:
        sampled_bpp_systems['redshift'] = np.where(sampled_bpp_systems['bin_num'] == bin_num, Z_redshifts[i], sampled_bpp_systems['redshift'])
        i += 1

    df = df.append(sampled_bpp_systems)
    initC_df = initC_df.append(sampled_initC_systems)


df.to_csv(save_path + "sampled_population_final.csv", index=False)
initC_df.to_csv(save_path + "sampled_initC_final.csv", index=False)    
