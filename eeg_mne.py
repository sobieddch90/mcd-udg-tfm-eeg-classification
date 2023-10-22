'''
EEG MNE Analysis
----------------
All the functions required to create new objects using Python MNE library, extract features and perform the ML modeling with EEG Data based on different approaches.
'''

# Author:     Elmo Chavez <sobieddchavez2022@gmail.com>
# Date:       October 8, 2023 

# default libraries
import sys
import os
import numpy as np
import pandas as pd

# Plot
import matplotlib.pyplot as plt
import seaborn as sns
# MNE Python for EEG
import mne
from mne.time_frequency import psd_array_welch #, tfr_morlet, tfr_multitaper
# Sklearn and Scipy utils
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.feature_selection import (SelectKBest, SelectPercentile,
                                       f_classif, mutual_info_classif, chi2)
from sklearn.model_selection import (KFold, StratifiedKFold,
                                     StratifiedShuffleSplit)
# Classifiers
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
import lightgbm as lgb
import xgboost as xgb

'''
Creating MNE Objects
--------------------
'''

def EEG_Raw_Data_List(path, subject_list=None):
    # Get the list of subfolders with all the EEG Datasets in the path
    path_items = os.listdir(path)
    # Get only subfolders
    subsjects = [item for item in path_items if os.path.isdir(os.path.join(path, item))]
    # Filter the subs selected
    if subject_list==None:
        subsjects_selected = subsjects
    elif set(subject_list).issubset(set(subsjects)):
        subsjects_selected = list(filter(lambda item: item in subject_list, subsjects))
    else:
        raise ValueError("List of Subjects is not valid, Please make sure you are using a valid list of Subjects.")
    subsjects_selected.sort()
    
    # Read EEG Raw Data for each subject
    list_eeg_raw = []
    for sub0XX in subsjects_selected:
        path_sub = path+'/'+sub0XX+'/eeg/'+sub0XX+'_task-eyesclosed_eeg.set'
        raw = mne.io.read_raw_eeglab(path_sub, preload=False, verbose='CRITICAL')
        list_eeg_raw.append(raw)
        
    print('EEG Raw Data readed:', len(list_eeg_raw))
    return list_eeg_raw

# ---
# Create MNE.Epochs with EEG Raw Data
def Epochs_Object(raw, channels=None, duration=60, overlapping=20, start_time=0.0, end_time=480):
    
    # Review channels
    if not ( channels == None or set(channels).issubset(set(raw.ch_names)) ) :
        raise ValueError("Channels parameter should be a valid list of Channels from the Raw Data or the keyword 'None'")
    
    # Create MNE.Epochs with a fixed length
    return mne.make_fixed_length_epochs(raw.copy().pick(channels).crop(tmin=start_time, tmax=end_time),
                                    duration=duration, overlap=overlapping, preload=True, verbose='CRITICAL')

# ---
# Create List of Epochs based on a list of Subjects (Required using the Dataset path)
def Epochs_Objects_MultiEEGs(path_dataset, subs_selected=None, channels=None, duration=60, overlapping=20, start_time=0.0, end_time=480):
    # Get the list of subfolders with all the EEG Datasets
    path_items = os.listdir(path_dataset)
    subjects = [item for item in path_items if os.path.isdir(os.path.join(path_dataset, item))]
    
    if not ( subs_selected == None or set(subs_selected).issubset(set(subjects)) ):
        raise ValueError("subs_selected should be a valid list or the keyword 'None'")
    
    # Create a list of EEG Epochs from Subjects Selected
    list_eeg_epochs = []
    for sub0XX in subs_selected:
        # Define path for sub0XX dataset
        path_sub = path_dataset+'/'+sub0XX+'/eeg/'+sub0XX+'_task-eyesclosed_eeg.set'
        # Create MNE.Raw
        raw = mne.io.read_raw_eeglab(path_sub, preload=False, verbose='CRITICAL')
        if not ( channels == None or set(channels).issubset(set(raw.ch_names)) ) :
            raise ValueError("Channels parameter should be a valid list of Channels from the Raw Data or the keyword 'None'")
        # Create Epochs
        epochs = mne.make_fixed_length_epochs(raw.copy().pick(channels).crop(tmin=start_time, tmax=end_time),
                                    duration=duration, overlap=overlapping, preload=True, verbose='CRITICAL')
        list_eeg_epochs.append(epochs)
    
    # Print summary
    print('EEG Raw Data readed:', len(list_eeg_epochs))
    print('Epochs (Windows) created for each EEG Data:', list_eeg_epochs[0].get_data().shape[0])
    # Return the list of epochs
    return list_eeg_epochs

'''
Feature Extraction
--------------------
'''

# Global variables
frequency_bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 45)
}

features = ['total_power', 
            'relative_power',
            'Std',
            'average_power',
            'spectral_entropy',
            'peak_to_peak']

# ---
# Create a Dictionary of features from PSD by using MNE.Epochs (Windows)
def PSD_Features_from_Epochs(epochs, sub_id=None, show_summary=True):
    # Set the variables
    sfreq = epochs.info['sfreq']
    n_channels = epochs.ch_names
    number_windows = epochs.get_data().shape[0]
    
    # Compute features from PSD method using PSD_ARRAY_WELCH
    results = {}
    for band, (fmin,fmax) in frequency_bands.items():
        # Apply bandpass filter to select only the frequency band range
        epochs_filtered = mne.filter.filter_data(epochs.get_data(), sfreq=sfreq, l_freq=fmin, h_freq=fmax, verbose='CRITICAL')
        
        # Compute the PSD using Welch's method
        psd_all, freqs_all = psd_array_welch(epochs.get_data(), sfreq=sfreq, verbose='CRITICAL')
        psd, freqs = psd_array_welch(epochs_filtered, sfreq=sfreq, verbose='CRITICAL')
        
        # Compute metrics & save results
        results[f'{band}_total_power'] = psd.sum(axis=2)
        results[f'{band}_relative_power'] = psd.sum(axis=2)/psd_all.sum(axis=2)
        results[f'{band}_std_dev'] = np.std(psd, axis=2)
        results[f'{band}_average_power'] = psd.mean(axis=2)
        results[f'{band}_spectral_entropy'] = -np.sum(np.log(psd) * psd, axis=2)
        results[f'{band}_peak_to_peak'] = np.ptp(psd,axis=2)

    flat_results = {}
    # Add Participant_ID
    if sub_id!=None:
        flat_results['participant_id'] = sub_id
    # Add features in a flat structure
    for key, array in results.items():
        for i in range(array.shape[0]):
            for j in range(array.shape[1]):
                new_key = f'w{i}_{n_channels[j]}_{key}'
                flat_results[new_key] = array[i, j]
    
    # Print Summary of results
    if show_summary:
        print('Summary...')
        print('-Number of Windows:', number_windows)
        print('-Number of Channels:', len(n_channels))
        print('-Frequency Bands:', len(frequency_bands))
        print('-Number of features computed:', len(features))
        print('--Total Features', len(flat_results))

    return flat_results

'''
ML Modeling
--------------------
'''

# Global variables
# ---

# Classifier Models
classifiers = {
    'Support Vector': SVC(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': xgb.XGBClassifier(),
    'LigthGBM': lgb.LGBMClassifier(),
    'AdaBoost': AdaBoostClassifier()
}

# Cross-Validation Method
cross_validation = { 
    'KFold': KFold(n_splits=5, shuffle=True, random_state=42),
    'StratifiedKFold': StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    'StratifiedShuffleSplit': StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
}

# Feature-Selection Method
feature_selector = {
    'anova': SelectKBest(score_func=f_classif),
    'mutual_info_classif': SelectKBest(score_func=mutual_info_classif),
    'chi2': SelectKBest(score_func=chi2)
}

# Results DataFrame columns
columns_results = [
    'feature_extraction',
    'channels',
    'classifier',
    'cross-validation',
    'feature-selection',
    'features_selected',
    'accuracy',
    'f1_score',
    'AUC'
]

# Classifier Model
def eeg_classifier_cv(df, feature_id, target, feature_extraction, channels):
    '''
    Execute a cross-validation prediction to classify classes by the following structure:
        - Classification through 5 different classifier models such as: SVC, Random Forest, XGBoost, LightGBM, and AdaBoost
        - Cross-validation through 3 different methods: KFold, StratifiedKFold, StratifiedShuffleSplit
        - Performe the prediction by whether or not a Feature Selection method is applied, such as: Anova, Chi2
        - Evaluate each classifier model by using metrics such as: Accuracy, F1_Score, AUC.
    
    Parameters:
    -----------
    df:         EEG dataset in a Pandas DataFrame object that contains the identify field and the target.
    feature_id: The respective identification field for each item.
    target:     The target that the model will try to predict.
    feature_extraction: For the output purpose this is a string required only to create the output DataFrame and stored as an identification.
    channels:   For the output purpose this is a string required only to create the output DataFrame and stored as the list of channels contained in the dataset.
    feature_selection: To apply a feature selection method to the training dataset and ensure having a better result to the classification.
    
    Output:
    -------
    Dataframe:
        A dataframe that will contain the summary of all the classifier employed with the following columns:
        feature_extraction, channels, classifier, cross-validation, feature-selection, features_selected, accuracy, f1_score, AUC
    '''
    
    # Define X, y
    X = df.drop(columns=[feature_id, target]).values
    features =  df.drop(columns=[target]).columns
    y = df['Group'].values
    
    # Run classifier model by using cross-validation, feature-selection and storing results.
    results = []
    for clf_name, clf in classifiers.items():
        print('Running:', clf_name)
        for cv_name, cv in cross_validation.items():
            for selector_nm, selector in feature_selector.items():
                accuracy_scores = []
                f1_scores = []
                auc_scores = []
                for train_index, test_index in cv.split(X, y):
                    # Split the data into training and testing sets based on the current fold indices
                    X_train, X_test = X[train_index], X[test_index]
                    y_train, y_test = y[train_index], y[test_index]
                    
                    # Feature Selection
                    X_train_selected = selector.fit_transform(X_train, y_train)
                    X_test_selected = selector.transform(X_test)
                    
                    # Train the classifier on the training data
                    clf.fit(X_train_selected, y_train)
                    
                    # Make predictions on the test data
                    y_pred = clf.predict(X_test_selected)
                    #y_score = clf.predict_proba(X_test_selected)
                    
                    # Calculate metrics for the current fold and append to the scores list
                    accuracy_scores.append(accuracy_score(y_true=y_test, y_pred=y_pred))
                    f1_scores.append(f1_score(y_true=y_test, y_pred=y_pred, average="macro"))
                    auc_scores.append(roc_auc_score(y_test, y_pred, multi_class="ovr"))
            
                # Save results for each option
                result = {'feature_extraction': feature_extraction,
                            'channels': channels,
                            'classifier': clf_name,
                            'cross-validation': cv_name,
                            'feature-selection': selector_nm,
                            'accuracy': np.mean(accuracy_scores),
                            'f1_score': np.mean(f1_scores),
                            'AUC': np.mean(auc_scores)
                            }
                
                results.append(result)
    
    df_results = pd.DataFrame(results)
    return df_results

'''
Other functions
--------------------
'''

def Dataset_Features_Summary(df):
    # Get list of columns
    list_columns = [x.replace('_',' ') for x in df.columns.to_list()]
    # Clean participants info
    list_columns_cleaned = [x for x in list_columns if x not in ['participant id','Age','Gender','Group']]
    # Get a list with all the features section
    windows = list(set([x.split()[0] for x in list_columns_cleaned]))
    windows.sort()
    channels = list(set([x.split()[1] for x in list_columns_cleaned]))
    freq_bands = list(set([x.split()[2] for x in list_columns_cleaned]))
    features = list(set([x[x.find(' ', 10):].strip() for x in list_columns_cleaned]))
    # Print list of features attributes
    print('Total Features:', len(list_columns))
    print('Windows:', len(windows), '->', windows)
    print('Channels:', len(channels),'->', channels)
    print('Frequency Bands:', len(freq_bands), '->', freq_bands)
    print('Features:', len(features),'->', features)