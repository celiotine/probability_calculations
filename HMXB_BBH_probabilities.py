import pandas as pd
import numpy as np
import glob
import sys
import re
from scipy import interpolate
from astropy.cosmology import Planck15 as cosmo
from astropy.cosmology import z_at_value
import astropy.units as u
from cosmic.evolve import Evolve
from cosmic.sample.initialbinarytable import InitialBinaryTable

#----------------------------------------------------------------------------------                                

## phsyical constants                                                                            
c = 2.99e10        ## speed of light in cm/s                                                  
secyr = 3.154e7    ## seconds per year                                                            
Myr = 1e6          ## years per Myr                                                            
Msun = 1.989e33    ## grams per solar mass                                                           
Lsun = 3.839e33    ## erg/sec per solar luminosity                                               

#-----------------------------------------------------------------------------------                    
## analytic approximation for P(omega) from Dominik et al. 2015                                                                 
def P_omega(omega_values):
    return 0.374222*(1-omega_values)**2 + 2.04216*(1-omega_values)**4 - 2.63948*(1-omega_values)**8 + 1.222543*(1-omega_values)**10
 
#----------------------------------------------------------------------------------                                                                  
### Monte Carlo sampling for detections above the given SNR threshold                                                          \
def calc_detection_prob(m1, m2, z_merge):

    ## constants that reflect LIGO design sensitivity                                                                           
    d_L8 = 1  ## in Gpc                                                                                                         
    M_8 = 10  ## in Msun                                                                                                       \
                                                                                                                                
    SNR_thresh = 8

    ## approximate typical SNR from Fishbach et al. 2018                                                                        
    M_chirp = (m1*m2)**(3./5)/(m1+m2)**(1./5)
    d_C = cosmo.comoving_distance(z_merge).to(u.Gpc).value
    d_L = (1+z_merge)*d_C

    rho_0 = 8*(M_chirp*(1+z_merge)/M_8)**(5./6)*d_L8/d_L   ## this is the "typical/optimal" SNR                                 
    if (rho_0 < SNR_thresh): return 0

    ## sample omega according to distribution for omega via inverse CDF method                                            
    dist_size = 10000
    sample_size = 1000
    P_omega_dist = P_omega(np.linspace(0, 1, dist_size))
    inv_P_omega = interpolate.interp1d(P_omega_dist, np.linspace(0, 1, dist_size), fill_value="extrapolate")
    omega = inv_P_omega(np.random.uniform(0, 1, sample_size))

    ## find the true SNRs given sky location                                                                                    
    rho = omega*rho_0
    accept_SNR_num = len(rho[np.where(rho >= SNR_thresh)])

    p_det = accept_SNR_num/sample_size

    return p_det

#-----------------------------------------------------------------------------------# 
## calculate XRB flux time series information
def calc_flux(current_BH_mass, initial_BH_mass, mdot_BH, d_L):

    bolometric_correction = 0.8
    where_lower_masses = current_BH_mass < np.sqrt(6)*initial_BH_mass

    eta_lower_masses = 1 - np.sqrt(1-(current_BH_mass/(3*initial_BH_mass))**2)
    eta = np.where(where_lower_masses, eta_lower_masses, 0.42)

    acc_rate = mdot_BH/(1-eta)  ## accretion rate in Msun/year
    luminosity = bolometric_correction*eta*acc_rate*c**2*Msun/secyr   ## accretion luminosity in erg/sec  
    flux = luminosity/(4 * np.pi * d_L**2)   ## flux in erg/s/cm^2

    return flux

#----------------------------------------------------------------------------------

columns=['bin_num', 'metallicity', 'merger_type', 'bin_state', 'delay_time', 'lookback_time', 'z_f', 'p_det', 'merge_by_z0', 'ZAMS_mass_k1','ZAMS_mass_k2', 'remnant_mass_k1', 'remnant_mass_k2', 'final_k1', 'final_k2', 'BH_mass_i', 'donor_mass_i', 'donor_type', 'XRB_sep_i', 'XRB_porb_i', 'emit11', 'emit13', 'emit15']
df_all = pd.DataFrame(columns=columns)

sampled_pop = pd.read_csv(sys.argv[1])  ## file of sampled population; file is structured like bpp array, csv format
sampled_initC = pd.read_csv(sys.argv[2])  ## file of sampled population initial conditions; file is structured like initCond file, csv format 

binary_IDs = np.unique(sampled_pop['bin_num'].values)

dtp = 0.01
timestep_conditions = [['kstar_1<13', 'kstar_2=14', 'dtp=0.01'], ['kstar_1=14', 'kstar_2<13', 'dtp=0.01']]

## initialize all counting variables (for probabilities) to zero
BBH_count = 0
BBHm_count = 0
BBHm_obs_count = 0

HMXB_count = 0
HMXB_obs_count = 0

HMXB_BBHm = 0
HMXB_obs_BBHm = 0

HMXB_BBH = 0
HMXB_obs_BBH = 0

HMXB_obs_BBHm_obs = 0

sample_size = 0

#----------------------------------------------------------------------------------

## loop through all binaries in the sample population, calculate XRB and DCO properties
for binary in binary_IDs:
    
    bpps = sampled_pop.loc[np.where(sampled_pop['bin_num'] == binary)]
    initCs = sampled_initC.loc[np.where(sampled_initC['bin_num'] == binary)]

    bpp_mets = np.unique(bpps['metallicity'])
    initC_mets = np.unique(initCs['metallicity'])
    bin_num = binary

    for bpp_met, initC_met  in zip(bpp_mets, initC_mets):
        
        sample_size += 1

        ## true/false params for the binary
        this_BBHm_obs = False
        this_BBHm = False
        this_BBH = False

        ## get the COSMIC information for this binary
        sample_bpp = bpps.iloc[np.where(bpps['metallicity'] == bpp_met)[0]]
        sample_initC = initCs.iloc[np.where(initCs['metallicity'] == initC_met)[0]]
        if (sample_initC.shape[0] > 1): sample_initC = sample_initC.iloc[:-1]

        ## re-evolve this binary to get small-timestep XRB info
        bpp, bcm, initC, kick_info = Evolve.evolve(initialbinarytable=sample_initC, timestep_conditions=timestep_conditions)

        merger_type = int(bcm['merger_type'].iloc[-1])
        bin_state = bcm['bin_state'].iloc[-1]

        z_f = sample_bpp['redshift'].iloc[-1]
        d_L = (1+z_f)*cosmo.comoving_distance(z_f).to(u.cm).value    ## luminosity distance, in cm for flux calculation

        ## get ZAMS masses for the binary                                                
        ZAMS_mass_k1 = bpp['mass_1'].iloc[0]
        ZAMS_mass_k2 = bpp['mass_2'].iloc[0]

        ## get final COSMIC merge types for the binary                                 
        final_k1 = bpp['kstar_1'].iloc[-1]
        final_k2 = bpp['kstar_2'].iloc[-1]
        
        ## count alive BBHs
        if (final_k1 == 14 and final_k2 == 14 and bin_state == 0):
            BBH_count += 1
            this_BBH = True
        #----------------------------------------------------------------------------------

        try: merge_index = np.where(bpp['evol_type']==6)[0][0]
        except: merge_index = -1

        ## CASE 1: system merges                                                                  
        ## assign lookback time w/ randomly sampled redshift from weighted distribution                
        ## alive/disrupted systems have merge_index = -1                                             
        if (merger_type != -1):

            ## count all merged BBHs in COSMIC
            if (merger_type == 1414):
                BBH_count += 1
                this_BBH = True

            remnant_mass_k1 = bpp['mass0_1'].iloc[merge_index]
            remnant_mass_k2 = bpp['mass0_2'].iloc[merge_index]

            delay_time = bpp['tphys'].iloc[merge_index]
            z_merge = z_at_value(cosmo.lookback_time, delay_time*u.Myr)

            lookback_time = cosmo.lookback_time(z_f).to(u.Myr).value

            if (delay_time <= lookback_time):
                merge_by_z0 = 1

                p_det = calc_detection_prob(remnant_mass_k1, remnant_mass_k2, z_merge) ## detection probability for this merger

                if (p_det > 0):
                    this_BBHm_obs = True
                    
                if (merger_type == 1414):
                    this_BBHm = True
                    BBHm_count += 1
                    BBHm_obs_count += p_det

            else: merge_by_z0 = 0

            remnant_mass_k1 = bpp['mass0_1'].iloc[merge_index]
            remnant_mass_k2 = bpp['mass0_2'].iloc[merge_index]

        ## CASE 2: system does not merge           
        ## set params to very small positive value for plotting purposes
        else:
            delay_time = 1e-8     
            lookback_time = 1e-8
            merge_by_z0 = 0
            remnant_mass_k1 = bpp['mass_1'].iloc[-1]
            remnant_mass_k2 = bpp['mass_2'].iloc[-1]
            p_det = 0
        #----------------------------------------------------------------------------------

        ## CASE A: system doesn't undergo an HMXB phae                                            
        ## if so, there are only 2 rows in bcm frame                      
        if (bcm.shape[0] < 3):
            BH_mass_i = -1
            donor_mass_i = -1
            donor_type = -1
            XRB_sep_i = -1
            XRB_porb_i = -1
            emit11 = 0
            emit13 = 0
            emit15 = 0

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


            XRB_sep_i = bcm['sep'].iloc[XRB_index]
            
            ## check if system is disrupted
            if (XRB_sep_i == -1):
                BH_mass_i = -1
                donor_mass_i = -1
                donor_type = -1
                XRB_sep_i = -1
                XRB_porb_i = -1
                emit11 = -1
                emit13 = -1
                emit15 = -1
            
            else:

                HMXB_count += 1
                if (this_BBH): HMXB_BBH += 1
                if (this_BBHm): HMXB_BBHm += 1


                ## get binary params at beginning of XRB phase                                 
                donor_mass_i = bcm[donorMass].iloc[XRB_index]
                donor_type = bcm[donorObj].iloc[XRB_index]
                BH_mass_i = bcm[BHmass].iloc[XRB_index]
                XRB_sep_i = bcm['sep'].iloc[XRB_index]
                XRB_porb_i = bcm['porb'].iloc[XRB_index]

                BH_mdot = bcm[BHmdot]  ## BH accretion data                                
                flux = calc_flux(bcm[BHmass], np.ones(len(bcm[BHmass]))*BH_mass_i, BH_mdot, d_L)
                emit15 = len(np.where(flux > 1e-15)[0])*dtp
                emit13 = len(np.where(flux > 1e-13)[0])*dtp
                emit11 = len(np.where(flux > 1e-11)[0])*dtp

                if (emit15 > 0): 
                    HMXB_obs_count += 1
                    if (this_BBH): HMXB_obs_BBH += 1
                    if (this_BBHm): HMXB_obs_BBHm += 1
                    if (this_BBHm_obs): HMXB_obs_BBHm_obs += p_det
            #----------------------------------------------------------------------------------

        df = pd.DataFrame([[bin_num, bpp_met, merger_type, bin_state, delay_time, lookback_time, z_f, p_det, merge_by_z0, ZAMS_mass_k1, ZAMS_mass_k2, remnant_mass_k1, remnant_mass_k2, final_k1, final_k2, BH_mass_i, donor_mass_i, donor_type, XRB_sep_i, XRB_porb_i, emit11, emit13, emit15]], columns=columns)

        df_all = df_all.append(df, sort=False, ignore_index=True)

df_all.to_csv("sampled_HMXB_params.csv", index=False)

print ("Sample Size:", sample_size)

print ("Probabilities:\np(BBH): ", BBH_count/sample_size, "\np(BBHm): ", BBHm_count/sample_size, "\np(BBHm_obs): ", BBHm_obs_count/sample_size)
print ("\np(HMXB): ", HMXB_count/sample_size, "\np(HMXB_obs): ", HMXB_obs_count/sample_size)

if (BBHm_count == 0): HMXB_BBHm_prob = 0
else: HMXB_BBHm_prob = HMXB_BBHm/BBHm_count

if (BBHm_obs_count == 0): HMXB_obs_BBHm_obs_prob = 0
else: HMXB_obs_BBHm_obs_prob = HMXB_obs_BBHm_obs/BBHm_obs_count

if (HMXB_count == 0): BBHm_HMXB_prob = 0
else: BBHm_HMXB_prob = HMXB_BBHm/HMXB_count

if (HMXB_obs_count == 0): 
    BBHm_HMXB_obs_prob = 0
    BBHm_obs_HMXB_obs_prob = 0
else: 
    BBHm_HMXB_obs_prob = HMXB_obs_BBHm/HMXB_obs_count
    BBHm_obs_HMXB_obs_prob = HMXB_obs_BBHm_obs/HMXB_obs_count


print ("\np(HMXB | BBHm): ", HMXB_BBHm_prob, "\np(HMXB_obs | BBHm_obs): ", HMXB_obs_BBHm_obs_prob) 
print ("\np(BBH | HMXB): ", HMXB_BBH/HMXB_count, "\np(BBH | HMXB_obs): ", HMXB_obs_BBH/HMXB_count)
print ("\np(BBHm | HMXB): ", BBHm_HMXB_prob, "\np(BBHm | HMXB_obs): ", BBHm_HMXB_obs_prob)
print ("\np(BBHm_obs | HMXB_obs): ", BBHm_obs_HMXB_obs_prob)
