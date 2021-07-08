import pandas as pd
import numpy as np
import sys
#---------------------------------------------------------------------------------- 

df = pd.read_csv(sys.argv[1])

sample_size = df.shape[0]
thresh = 'emit15'

BBH_obj = df.iloc[np.where(df['this_BBH'])]
BBHm_obj = df.iloc[np.where(df['this_BBHm'])]
HMXB_obj = df.iloc[np.where(df['this_HMXB'])]
HMXB_obs_obj = HMXB_obj.iloc[np.where(HMXB_obj[thresh] > 0)]

BBH_HMXB_obj = BBH_obj.iloc[np.where(BBH_obj['this_HMXB'])]
BBHm_HMXB_obj = BBHm_obj.iloc[np.where(BBHm_obj['this_HMXB'])]

## save probability counts
BBH_count = BBH_obj.shape[0]
BBHm_count = BBHm_obj.shape[0]
HMXB_count = HMXB_obj.shape[0]

HMXB_obs_count = HMXB_obs_obj.shape[0]
BBHm_obs_count = np.sum(BBHm_obj['p_det'])

HMXB_BBH_count = BBH_HMXB_obj.shape[0]
HMXB_BBHm_count = BBHm_HMXB_obj.shape[0]
HMXBobs_BBHmobs_count = np.sum(BBHm_HMXB_obj.iloc[np.where(BBHm_HMXB_obj[thresh] > 0)]['p_det'])
HMXBobs_BBH_count = len(np.where(BBH_obj[thresh] > 0)[0])
HMXBobs_BBHm_count = len(np.where(BBHm_obj[thresh] > 0)[0])

## calculate non-conditional probabilities
pBBH = BBH_count/sample_size
pBBHm = BBHm_count/sample_size
pBBHm_obs = BBHm_obs_count/sample_size

pHMXB = HMXB_count/sample_size
pHMXB_obs = HMXB_obs_count/sample_size

## p(HMXB | BBHm)
pHMXB_BBHm = HMXB_BBHm_count/BBHm_count

## p(HMXB_obs | BBHm_obs)
pHMXBobs_BBHmobs = HMXBobs_BBHmobs_count/BBHm_obs_count

## f(HMXB_obs | BBHm_obs)
fHMXBobs_BBHmobs = np.sum(BBHm_HMXB_obj[thresh] * BBHm_HMXB_obj['p_det'])/np.sum(BBHm_HMXB_obj['emit_tot']* BBHm_HMXB_obj['p_det'])

## p(BBH | HMXB)
pBBH_HMXB = HMXB_BBH_count/HMXB_count

## f(BBH | HMXB)
fBBH_HMXB = np.sum(BBH_HMXB_obj['emit_tot'])/np.sum(HMXB_obj['emit_tot'])

## p(BBH | HMXB_obs)
pBBH_HMXBobs = HMXBobs_BBH_count/HMXB_obs_count

## f(BBH | HMXB_obs)
fBBH_HMXBobs = np.sum(BBH_HMXB_obj[thresh])/np.sum(HMXB_obj[thresh])

## p(BBHm | HMXB)
pBBHm_HMXB = HMXB_BBHm_count/HMXB_count

## f(BBHm | HMXB)
fBBHm_HMXB = np.sum(BBHm_HMXB_obj['emit_tot'])/np.sum(HMXB_obj['emit_tot'])

## p(BBHm | HMXB_obs)
pBBHm_HMXBobs = HMXBobs_BBHm_count/HMXB_obs_count

## f(BBHm | HMXB_obs)
fBBHm_HMXBobs = np.sum(BBHm_HMXB_obj[thresh])/np.sum(HMXB_obj[thresh])

## p(BBHm_obs | HMXB_obs)
pBBHmobs_HMXBobs = HMXBobs_BBHmobs_count/HMXB_obs_count

## f(BBHm_obs | HMXB_obs)
fBBHmobs_HMXBobs = np.sum(BBHm_HMXB_obj[thresh] * BBHm_HMXB_obj['p_det'])/np.sum(HMXB_obj[thresh])


print ("Probabilities:\np(BBH): ", pBBH, "\np(BBHm): ", pBBHm, "\np(BBHm_obs): ", pBBHm_obs)
print ("\np(HMXB): ", pHMXB, "\np(HMXB_obs): ", pHMXB_obs)

print ("\np(HMXB | BBHm): ", pHMXB_BBHm, "\np(HMXB_obs | BBHm_obs): ", pHMXBobs_BBHmobs)
print ("\np(BBH | HMXB): ", pBBH_HMXB, "\np(BBH | HMXB_obs): ", pBBH_HMXBobs)
print ("\np(BBHm | HMXB): ", pBBHm_HMXB, "\np(BBHm | HMXB_obs): ", pBBHm_HMXBobs)
print ("\np(BBHm_obs | HMXB_obs): ", pBBHmobs_HMXBobs)

print("\n------------------------------------------------------------------------")

print("\nTime Fractions: ", "\nf(HMXB_obs | BBHm_obs): ", fHMXBobs_BBHmobs)
print("\nf(BBH | HMXB): ", fBBH_HMXB, "\nf(BBH | HMXB_obs): ", fBBH_HMXBobs)
print("\nf(BBHm | HMXB): ", fBBHm_HMXB, "\nf(BBHm | HMXB_obs): ", fBBHm_HMXBobs)
print("\nf(BBHm_obs | HMXB_obs): ", fBBHmobs_HMXBobs)
