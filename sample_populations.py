import pandas as pd
import numpy as np
import scipy
import glob
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

all_COSMIC_runs = glob.glob(population_path)

Zsun = 0.017
lowZ = Zsun/200 #lower bound on metallicity                              
highZ = Zsun #upper bound on metallicity                                                     
sigmaZ = 0.5 #sigma of the lognormal distribution about the mean metallicity                            

bpp_pops = []
bbh_pops = []
mets = []

## read-in all COSMIC populations                                                                              
for COSMIC_run in all_COSMIC_runs:
    ## read bpp data
    data = glob.glob(COSMIC_run + "/*.h5")[0]
    bpp = pd.read_hdf(data, key='bpp')
    bpp_pops.append(bpp)

    met_val = float(re.findall('\d+.\d+', COSMIC_run)[0])
    met_format_str = "{:.8f}".format(met_val)
    mets.append(float(met_format_str))  ## run names are the population metallicity

    #num_systems = len(np.unique(bpp.index.values))

    ## find BBH systems in bpp array                                                               
    #bbh_idxs = bpp.loc[(bpp['kstar_1']==14) & (bpp['kstar_2']==14)].index.unique()
    #bpp_cut = bpp.loc[bbh_idxs]
    #bbh_mergers  = bpp_cut.loc[(bpp_cut['evol_type']==6)]
    #bbh_pops.append(bbh_mergers)

print (mets)

print ("starting sampling...")

#num_mergers = 0
#sampled_Z = []

## grid of BBH formation redshifts                                      
zf_grid = np.linspace(0, 20, 1000)
SFR = rates.sfr_z(zf_grid, mdl='2017')
dVdz = cosmo.differential_comoving_volume(zf_grid)
zf_weights = SFR * dVdz * (1+zf_grid)**-1

zf_picks = np.random.choice(zf_grid, N, replace=True, p=zf_weights/np.sum(zf_weights))

## sample a metallicity for each sampled formation redshift 
df = pd.DataFrame()

for zf_pick in zf_picks:
    
    ### get metallicity weights for all drawn formation redshifts                                              
    Z_weights = functions.metal_disp_z(zf_pick, np.array(mets), sigmaZ, lowZ, highZ)

    ## draw metallicity using the above weights to choose a COSMIC population to draw a system from             
    Z_pick = np.random.choice(mets, p=Z_weights/np.sum(Z_weights))
    #sampled_Z.append(Z_pick)

    pop_idx = np.where(mets==Z_pick)[0][0]
    this_pop = bpp_pops[pop_idx]   ## the population of DCOs from which to draw a system; this is the full bpp file from that COSMIC run
    ids = np.unique(this_pop['bin_num'].values)
    
    this_binary = np.random.choice(ids, replace=True)  ## the sampled binary from all COSMIC populations
    this_pop.loc[this_binary, "metallicity"] = Z_pick
    this_pop.loc[this_binary, "redshift"] = zf_pick
    
    df = df.append(this_pop.loc[this_binary])

    #this_bbh_pop = bbh_pops[pop_idx]
    #bbh_ids = np.unique(this_bbh_pop['bin_num'].values)

df.to_csv("sampled_population.csv", index=False)    
