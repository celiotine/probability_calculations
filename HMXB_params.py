import pandas as pd
import numpy as np
import glob
import sys
import re
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

#----------------------------------------------------------------------------------                                

## phsyical constants                                                                            
c = 2.99e10        ## speed of light in cm/s                                                  
secyr = 3.154e7    ## seconds per year                                                            
Myr = 1e6          ## years per Myr                                                            
Msun = 1.989e33    ## grams per solar mass                                                           
Lsun = 3.839e33    ## erg/sec per solar luminosity                                               
 
#----------------------------------------------------------------------------------                         

def calc_flux(current_BH_mass, initial_BH_mass, mdot_BH, d_L):

    bolometric_correction = 0.8
    where_lower_masses = current_BH_mass < np.sqrt(6)*initial_BH_mass

    eta_lower_masses = 1 - np.sqrt(1-(current_BH_mass/(3*initial_BH_mass))**2)
    eta = np.where(where_lower_masses, eta_lower_masses, 0.42)

    acc_rate = mdot_BH/(1-eta)
    luminosity = bolometric_correction*eta*acc_rate*c**2*Msun/secyr   ## accretion luminosity in erg/sec  
    flux = luminosity/(4 * np.pi * d_L**2)

    return flux

#----------------------------------------------------------------------------------

columns=['bin_num', 'merger_type', 'bin_state', 'delay_time', 'lookback_time', 'z_f', 'merge_by_z0', 'ZAMS_mass_k1','ZAMS_mass_k2', 'remnant_mass_k1', 'remnant_mass_k2', 'final_k1', 'final_k2', 'BH_mass_i', 'donor_mass_i', 'donor_type', 'XRB_sep_i', 'XRB_porb_i', 'emit11', 'emit13', 'emit15']
df_all = pd.DataFrame(columns=columns)

population_file = sys.argv[1]  ## file of sampled population; file is structured like bpp array, csv format
COSMIC_runs = sys.argv[2]  ## path to directory of COSMIC runs for all metallicities; need this for small-dtp tracks for XRBs

sampled_pop = pd.read_csv(population_file)

binary_IDs = np.unique(sampled_pop['bin_num'].values)
dtp = 0.01

#----------------------------------------------------------------------------------

for binary in binary_IDs:
    
    print (binary)
    bpps = sampled_pop.loc[np.where(sampled_pop['bin_num'] == binary)]

    #print (bpp['metallicity'])
    mets = np.unique(bpps['metallicity'])
    bin_num = binary
    #print (mets)

    for met in mets:
        
        ## dumb string formatting shit
        format_met_string = "{:.8f}".format(met)
        met = float(format_met_string)
        #print (format_met_string)

        #print (np.where(bpps['metallicity'] ==met)[0])
        bpp = bpps.iloc[np.where(bpps['metallicity'] == met)[0]]
        print (met)
        print (bpp)
        #print (met)

        bcm = pd.read_csv(COSMIC_runs + format_met_string + "/evolved_tracks/" + str(binary) + "_bcm.csv")

    
        merger_type = int(bcm['merger_type'].iloc[-1])
        bin_state = bcm['bin_state'].iloc[-1]

        z_f = bpp['redshift'].iloc[-1]
        #print (z_f)
        d_L = (1+z_f)*cosmo.comoving_distance(z_f).to(u.cm).value    ## luminosity distance, in cm for flux calculation

        ## get ZAMS masses for the binary                                                
        ZAMS_mass_k1 = bpp['mass_1'].iloc[0]
        ZAMS_mass_k2 = bpp['mass_2'].iloc[0]

        ## get final COSMIC merge types for the binary                                 
        final_k1 = bpp['kstar_1'].iloc[-1]
        final_k2 = bpp['kstar_2'].iloc[-1]


        #----------------------------------------------------------------------------------

        try: merge_index = np.where(bpp['evol_type']==6)[0][0]
        except: merge_index = -1

        ## CASE 1: system merges                                                                  
        ## assign lookback time w/ randomly sampled redshift from weighted distribution                
        ## alive/disrupted systems have merge_index = -1                                             
        if (merge_index != -1):
            delay_time = bpp['tphys'].iloc[merge_index]

            lookback_time = cosmo.lookback_time(z_f).to(u.Myr).value

            if (delay_time <= lookback_time): merge_by_z0 = 1
            else: merge_by_z0 = 0

            remnant_mass_k1 = bpp['mass0_1'].iloc[merge_index]
            remnant_mass_k2 = bpp['mass0_2'].iloc[merge_index]

        ## CASE 2: system does not merge           
        ## set params to very small positive value for plotting purposes
        else:
            delay_time = 1e-8     
            lookback_time = 1e-8
            merge_by_z0 = 1e-8
            d_L = 1e-8
            remnant_mass_k1 = bpp['mass_1'].iloc[-1]
            remnant_mass_k2 = bpp['mass_2'].iloc[-1]

        #----------------------------------------------------------------------------------
    
        ## CASE A: system doesn't undergo an HMXB phae                                            
        ## if so, there are only 2 rows in bcm frame                      
        if (bcm.shape[0] < 3):
            BH_mass_i = -1
            donor_mass_i = -1
            donor_type = -1
            XRB_sep_i = -1
            XRB_porb_i = -1
            emit11 = -1
            emit13 = -1
            emit15 = -1

        ### CASE B: system undeoges an HMXB phase                                      
        else:
            ## get bcm index where each BH is formed                                   
            try: BH1_index = np.where(bcm['kstar_1'] == 14)[0][0]
            except: BH1_index = np.infty

            try: BH2_index = np.where(bcm['kstar_2'] == 14)[0][0]
            except: BH2_index = np.infty


            ## CASE Bi: BH1 (kstar_1) is formed first                                
            if (BH2_index > BH1_index):
                XRB_index = BH1_index
                BHobj = "kstar_1"
                donorObj = "kstar_2"
                BHmass = "mass_1"
                donorMass = "mass_2"
                BHmdot = "deltam_1"


            # CASE Bii: BH2 (kstar_2) is formed first                                  
            else:
                XRB_index = BH2_index
                BHobj = "kstar_2"
                donorObj = "kstar_1"
                BHmass = "mass_2"
                donorMass = "mass_1"
                BHmdot = "deltam_2"


            ## get binary params at beginning of XRB phase                                 
            donor_mass_i = bcm[donorMass][XRB_index]
            donor_type = bcm[donorObj][XRB_index]
            BH_mass_i = bcm[BHobj][XRB_index]
            XRB_sep_i = bcm['sep'][XRB_index]
            XRB_porb_i = bcm['porb'][XRB_index]

            BH_mdot = bcm[BHmdot]  ## BH accretion data                                
            flux = calc_flux(bcm[BHmass], np.ones(len(bcm[BHmass]))*BH_mass_i, BH_mdot, d_L)
            emit15 = len(np.where(flux > 1e-15)[0])*dtp
            emit13 = len(np.where(flux > 1e-13)[0])*dtp
            emit11 = len(np.where(flux > 1e-11)[0])*dtp

            #----------------------------------------------------------------------------------

            df = pd.DataFrame([[bin_num, merger_type, bin_state, delay_time, lookback_time, z_f, merge_by_z0, ZAMS_mass_k1, ZAMS_mass_k2, remnant_mass_k1, remnant_mass_k2, final_k1, final_k2, BH_mass_i, donor_mass_i, donor_type, XRB_sep_i, XRB_porb_i, emit11, emit13, emit15]], columns=columns)

    df_all = df_all.append(df, sort=False, ignore_index=True)

df_all.to_csv("sampled_HMXB_params.csv", index=False)
