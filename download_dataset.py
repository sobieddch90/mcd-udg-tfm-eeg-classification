'''
Author: Elmo Chavez
Date:   17-Jul-2023
---------------------
Use this script to download the Open Neuro EEG Dataset with id ds004504 in your local machine.
We recommend run this script only if you already have previosly installed the openneuro package.
'''

import os
import sys

import openneuro as on

print('------ Downloading EEG Dataset ------')
print('Where do you want to store the dataset? Make sure you select the correct directory. We suggest selecting an empty folder.')
directory_path = input(' Please enter the directory Path:')

if os.path.exists(directory_path):
    os.chdir(directory_path)
    print("Directory selected:", directory_path)
    
    if len(os.listdir(directory_path)) == 0:
        print('The directory is empty')
    else:
        option = input('The directory is not empty. Do you want to download the Dataset here? (y/n)')
        if str.lower(option) == 'y':
            print('Downloading')
        else:
            sys.exit("Downloading cancelled")
    
    dataset_id = "ds004504"
    on.download(dataset=dataset_id, target_dir=directory_path)
else:
    print("Directory does not exist.")
