import pandas as pd
import glob

## concatenate data frames into one

path = "HMXB_output/*"
all_param_files = glob.glob(path)

#df = pd.read_csv("./0_params.csv")
df = pd.DataFrame()

for pfile in all_param_files:

    #if pfile == "0_params.csv": continue

    pf = pd.read_csv(pfile)
    df = df.append(pf, ignore_index=True)
    
    
df.to_csv("all_xrb_parameters.csv", index=False)
