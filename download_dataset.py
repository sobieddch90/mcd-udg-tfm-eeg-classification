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
dataset_id = 'ds004504'

try:
    # Creating directory for the dataset
    print("Creating directory...")
    os.mkdir(dataset_id)
    print()
    
    # Downloading the dataset
    dataset_path = os.getcwd()+"/"+dataset_id
    on.download(dataset=dataset_id, target_dir=dataset_path)
    
except:
  print("Something went wrong, Please review the directory.")
