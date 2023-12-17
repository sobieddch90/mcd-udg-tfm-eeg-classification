'''
Consolidate Results
----------------
Consolidate the results obtained from all the cross validation methods developed during the project.
'''
import pandas as pd
import os

# Results files
print('--Initiating--')
path = os.getcwd()
files = ['Results PSD - Cross-Validation.csv',
         'Results PSD Alpha - Cross-Validation.csv',
         'Results PSD Total Power & Spectral Density - Cross-Validation.csv',
         'Results PSD no Freq-Bands - Cross-Validation.csv']

# Read all files
print('--Reading files--')
df_list = []
for file in files:
    df = pd.read_csv(path+"/Training Dataset/"+file, index_col=0)
    df_list.append(df)
    print('File readed:', file)

# Concatenate all files
print('--Concatenating results--')
df_results = pd.concat(df_list, ignore_index=True)

# Save results
print('--Saving Results--')
filename = path+'/Training Dataset/All Results PSD.csv'
df_results.to_csv(filename, index=False)
print('--Completed--')
