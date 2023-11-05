'''
EEG MNE Analysis
----------------
All the functions required to create new objects using Python MNE library, extract features and perform the ML modeling
with EEG Data based on different approaches.
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

# Sklearn utils
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
-------------------------------------------------------------
Creating MNE Objects
-------------------------------------------------------------
'''

def EEG_Raw_Data_List(path, subject_list=None):
    '''
    Create a list of MNE.Raw data objects from the ds004504 data source, the process will go through all the directories 
    and read all the .set objects and creating an MNE.Raw object. 
    
    Parameters:
    -----------
    path:   String
        The path that contains all the EEG Datasets with a `.set` file format, the process to read will go through all the 
        directories and creating an MNE.Raw object. 
    subject_list: List of Strings
        With the aim of only read a subset of the EEG Datasets from all contained in the dataset, you can provide a list of
        string with the name of objects you want to read.
    
    Output:
    -----------
    List:   List of MNE.Raw objects
    '''
    
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
    '''
    From an MNE.raw object, the current function will create a fixed epochs (windows) from the raw, each windows can have a 
    fixed duration and fixed overlaping provided in the parameters, and also the Raw data can be croped to have entired windows
    as a result.
    
    Parameters:
    -----------
    raw:   MNE.Raw
        It is required providing an MNE.Raw object to be allowed to use the MNE Library and create the MNE.Epochs (Windows).
    channels: List of Strings
        If the result of the MNE.Epochs requested should be from one or more specific EEG Channel, a list of strings that contains
        all the channels can be provided as a parameter. If the none value is provided the Epochs will be created by using all the 
        channels.
        Default value is None.
    duration:   integer
        Provide an integer between 10 and up to a number that is within the range of the time duration in the Raw data.
        Default value is 60 seconds.
    overalapping:   integer
        Provide an integer between 0 and up to the duration provided in the previous parameter.
        Default value is 20 seconds.
    start_time: integer
        The start time to filter the raw data before creating the MNE.Epochs.
    end_time:   integer
        The end time to filter the raw data before creating the MNE.Epochs.

    Output:
    -----------
    Epochs: MNE.Epoch object
        An MNE.Epoch object that is contains all the windows created from the parameters provided, all the windows are created 
        with the same structure (duration and overlapping) and since the start and end time of the raw data are part of the process
        the number of epochs will be the same for every raw data that is created with the same parameters.
    '''
    # Review channels
    if not ( channels == None or set(channels).issubset(set(raw.ch_names)) ) :
        raise ValueError("Channels parameter should be a valid list of Channels from the Raw Data or the keyword 'None'")
    
    # Create MNE.Epochs with a fixed length
    return mne.make_fixed_length_epochs(raw.copy().pick(channels).crop(tmin=start_time, tmax=end_time),
                                    duration=duration, overlap=overlapping, preload=True, verbose='CRITICAL')

# ---
# Create List of Epochs based on a list of Subjects (Required using the Dataset path)
def Epochs_Objects_MultiEEGs(path_dataset, subs_selected, channels=None, duration=60, overlapping=20, start_time=0.0, end_time=480):
    '''
    This function will run through the path provided and find all the eeg datasets with the extension `.set` and the create 
    the MNE.raw object first to subsequent create the MNE.epoch with fixed windows by using the other parameters to customize
    the duration, overlapping and additionally the range of time in the raw data.
    
    Parameters:
    -----------
    path_dataset:   string
        A string that contains the path from the `ds004504` directory where the method will read all the EEG Datasets. It can be provided 
        the `ds004504` that contains the unpreprocessed EEG files or the `ds004504/derivatives` that have all the EEG files with the data
        previously preprocessed.
    subs_selected:  list of strings
        A list of strings with all the subjects that you want to get the MNE.Epoch.
    channels:       list of strings
        A list of strings with all the EEG Channels that you want to include in the MNE.Epoch.
    duration:       integer
        The duration in seconds to create each window in the MNE.Epoch object. 
    overlapping:    integer
        The Overlapping in seconds for each window in the MNE.Epoch object.
    start_time:     integer
        The initial time where the raw data will start the windows in the MNE.Epoch object.
    end_time:       end_time
        The end of the raw data where the MNE.Epoch object will stop creating the last window.
    Output:
    -----------
    list:   list of MNE.Epochs objects
        For each EEG file found in the path and selected through the parameters, a MNE.Epochs object will be created and stored in a list 
        to finally return this list as an output.
    '''
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
-------------------------------------------------------------
Feature Extraction
-------------------------------------------------------------
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
    '''
    From the MNE methods is possible to create a PSD instance for each window that can be used to obtain multiple features
    such as total power, standard deviation, average power, peak to peak, or spectral entropy, but due to the results 
    being split into the different frequency bands, the relative power is another important feature that can be added 
    to this important list of features.
    
    Parameters:
    -----------
    epochs:     MNE.Epoch object
        To get all the features for each window, it is required to provide an MNE.Epoch object that can be used to perform
        and create all the features.
    sub_id:     string
        Tag the result with the subject_id from the dataset source, this is an additional feature to ensure it doesn't blend 
        the results with that one obtained from another subject.
    show_summary:   Boolean
        An additional option to print a summary of features and data results.
    Output:
    -----------
    df:     pd.DataFrame
        A pandas DataFrame that contains the EEG signals for each frequency band for the entired Raw data.
    '''
    
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

# ---
# Create a Dictionary of features from PSD by using MNE.Epochs (Windows)
def Raw_by_Freq_Bands(raw, tmax=None, sub_id=None):
    '''
    From an EEG raw data as an MNE.Raw object, create a dataset that is splitted into the different frequency bands and stored in a DataFrame.
    
    Parameters:
    -----------
    raw:    MNE.Raw Object
        An MNE.Raw object that contains the signals recorded that can be filtered by each frequency band range.
        
    Output:
    -----------
    DataFrame:  pd.DataFrame
        A dataframe that contains the raw data separated by all the frequency bands (Delta, Theta, Alpha, Beta, Gamma) as an additional feature.
    '''
    # Create a list 
    frequency_map = list()
    if tmax==None:
        time_max = 480
    else:
        time_max = tmax
    
    for band, (fmin,fmax) in frequency_bands.items():
        # filter the raw data with the freq band range
        raw_band = raw.copy().crop(tmin=0.0, tmax=time_max).filter(fmin, fmax, n_jobs=None, l_trans_bandwidth=0.5, h_trans_bandwidth=1, verbose='CRITICAL')
        # create a DataFrame
        df = raw_band.to_data_frame()
        # add the freq band as a feature
        df['Frequency Band'] = band
        # append the df to the list
        frequency_map.append(df)

    # Convert the list of dataframes to an unique DataFrame
    df_result = pd.concat(frequency_map, ignore_index=True)
    # Add the subject id
    if sub_id != None:
        df_result['participant_id'] = sub_id
    
    return df_result


'''
-------------------------------------------------------------
ML Modeling
-------------------------------------------------------------
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
    df: DataFrame
        EEG dataset in a Pandas DataFrame object that contains the identify field and the target.
        feature_id: The respective identification field for each item.
    feature_id: String
        The name of the identifier column in the DataFrame sent.
    target: String
        The target that the model will try to predict, or the column name in the DataFrame sent as a df.
        feature_extraction: For the output purpose this is a string required only to create the output DataFrame and stored as an identification.
    feature_extraction: String
        Name of the Feature Extraction method used to get the features.
    channels:   List of Strings
        For the output purpose this is a list of strings with the channels in the dataset, required only to create the output DataFrame and 
        stored as the list of channels contained in the dataset.
    
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

# Get the Score Metrics from the Predictions
def get_Scores(y_test, y_predict, _print=False):
    '''
    A simple function that prints all the three score metrics (Accuracy, F1_Score, Auc) to evaluate the classifier model.
    
    Parameters:
    -----------
    y_test:     array, list or a pd.Series 
        The correct values from the Test data to evaluate the prediction.
    y_predict:  array, list or a pd.Series 
        The correct values from the Test data to evaluate the prediction.
    _print:     Boolean
        A boolean value, with True if you want to print the results

    Output:
    -----------
    scores:     dictionary
        All the thre score metrics evaluated to stored as a dictionary.
    '''
    
    scores = {}
    scores['Accuracy'] = accuracy_score(y_true=y_test, y_pred=y_predict)
    scores['F1 Score'] = f1_score(y_true=y_test, y_pred=y_predict, average="macro")
    scores['AUC'] = roc_auc_score(y_test, y_predict, multi_class="ovr")
    
    print('-- SCORES --')
    if _print:
        for sc_nm, sc_val in scores.items():
            print(f'{sc_nm}: {np.round(sc_val*100,2)} %')
    
    return scores

'''
-------------------------------------------------------------
Other functions
-------------------------------------------------------------
'''

# Get A Summary from the Training Dataset
def Dataset_Features_Summary(df):
    '''
    A simple function that prints a summary from the Feature Extraction datasets.
    
    Parameters:
    -----------
    df:     pd.DataFrame
        A pandas DataFrame obtained from the Feature Extraction step in the project.

    Output:
    -----------
        None
    '''
    
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

# Plot the Feature Importance and get a DataFrame.
def get_Feature_Importance(feature_names, importance_values, top_n=10):
    '''
    Create a pandas DataFrame that contains all the feature importance obtanied from the prediction.
    
    Parameters:
    -----------
    feature_names:      List of strings
        A list with all the features to create the DataFrame.
    importance_values:  List of values
        A list with all the importance values obtained from the classifier model.
    top_n:  Integer
        An integer to select only the N features and values.

    Output:
    -----------
    df:     pd.DataFrame
        A DataFrame that contain all the top N feature importance.
    '''
    
    # Create a DataFrame and Sort the values
    df = pd.DataFrame({'Feature': feature_names, 'Importance': importance_values})
    df.sort_values('Importance', ascending=False, inplace=True)
    # Get the top N features
    df_top_n = df.head(top_n)
    df_top_n = df_top_n[::-1]
    
    # Plot a bar chart
    plt.figure(figsize=(10, 6))
    bar = plt.barh(df_top_n['Feature'], df_top_n['Importance'])
    plt.xlabel('Importance')
    plt.ylabel('Feature')
    plt.title(f'Top {top_n} - Feature Importances')
    # Add data labels
    for rect in bar:
        width = rect.get_width()
        plt.text(width, rect.get_y() + rect.get_height() / 2, f'{width*100:.2f}%', ha='left', va='center')
    plt.show
    
    return df_top_n