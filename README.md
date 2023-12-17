
# Predictive Model of Dementia Disorders by using Frontal EEG Channels

## Summary
This project represents the final requirement for the Master's Degree in Data Science at the University of Girona, Catalonia, Spain. The general objective is to build a predictive model, leveraging a publicly accessible dataset sourced from the Open Neuro platform. This dataset contains EEG data gathered from individuals diagnosed with Alzheimer's disease and frontotemporal Dementia as well as healthy control subjects.

The predictive model includes two distinct components. The first facet involves classification exclusively employing the frontal EEG channels, while the second facet extends the scope to predict disorders utilizing all available EEG channels. A key aspect of this project involves comparing outcomes from both components. This comparison is vital in explaining the significance of the frontal channels, providing valuable insights for future research developed at the university with different BCI devices that only offer data from these channels.

**Author**:         Elmo Chavez\
**Date**:           September 30, 2023

<!-- TABLE OF CONTENTS -->
## Table of Content

- [Predictive Model of Dementia Disorders by using Frontal EEG Channels](#predictive-model-of-dementia-disorders-by-using-frontal-eeg-channels)
  - [Summary](#summary)
  - [Table of Content](#table-of-content)
  - [1. Planning and Methodology](#1-planning-and-methodology)
  - [2. Installation and Setup](#2-installation-and-setup)
  - [3. Data](#3-data)
  - [4. Methodological Contribution](#4-methodological-contribution)
    - [EEG Data Exploration](#eeg-data-exploration)
    - [Feature Extraction](#feature-extraction)
    - [Modeling](#modeling)
    - [Results](#results)
  - [5. Results](#5-results)

<!-- END OF TABLE OF CONTENTS -->

## 1. Planning and Methodology
The project's strategic plan is divided into the steps shown in the picture below.

<img src="Other resources/TFM workflow.png" alt="Project Stages"/>

\
Each step is explained in detail in the following sections of this document. You will find all the necessary technical requirements to reproduce this repository successfully.

## 2. Installation and Setup

**Create a Python Environment and Install Libraries**\
It is recommended to create a new python environment to configure the necessary libraries to execute the code successfully, this is possible through two possible ways:
1. Using the Create Environment command in VSCode: `> Python: Create Environment`, will create the Python environment internally in the project, and using the `requirements.txt` file will install all the libraries needed to run the code.
2. Create a Python virtual environment from scratch and install all the libraries used for this project:
   1. Create a Python environment called ".venv": `python3 -m venv .venv`
   2. Activate the environment: `. .venv/bin/activate`
   3. Install all the necessary libraries:
      - `pip install requests`   
      - `pip install os-sys`  
      - `pip install numpy`
      - `pip install pandas`
      - `pip install matplotlib`
      - `pip install seaborn`
      - `pip install -U scikit-learn`
      - `pip install scipy`
      - `pip install xgboost`
      - `pip install lightgbm` (MacOS device requires running: `brew install libomp`)
      - `pip install openneuro-py`
      - `pip install mne`

> In the following link, you will find a complete guideline to install the Python virtual environment, whether you are following the first or second method: [Python Environments in VScode](https://code.visualstudio.com/docs/python/environments)

> I highly recommend using the first option because it can save you time and ensure proper installation of dependencies.

## 3. Data

**Description:**\
The dataset contains the EEG resting state-closed eyes recordings from 88 subjects in total.\
Participants:
- 36 of them were diagnosed with Alzheimer's disease (AD group)
- 23 were diagnosed with Frontotemporal Dementia (FTD group)
- 29 were healthy subjects (CN group).

Cognitive and neuropsychological state was evaluated by the international Mini-Mental State Examination (MMSE). MMSE score ranges from 0 to 30, with lower MMSE indicating more severe cognitive decline.

The duration of the disease was measured in months and the median value was 25 with IQR range (Q1-Q3) being 24 - 28.5 months. Concerning the AD groups, no dementia-related comorbidities have been reported.

The average MMSE was:
- For the AD group was 17.75 (sd=4.5)
- For the FTD group was 22.17 (sd=8.22)
- For the CN group was 30.

The mean age:
- AD group was 66.4 (sd=7.9)
- FTD group was 63.6 (sd=8.2)
- CN group was 67.9 (sd=5.4).

Source Dataset:
[Open Neuro: Alzheimer's disease, Frontotemporal dementia and Healthy subjects](https://openneuro.org/datasets/ds004504/versions/1.0.5)

**Download the dataset:**\
To download this dataset you can run the following command in the terminal:\
`python3 download_dataset.py`\
This will execute a python script to download the whole dataset within a new directory named `/ds004504`

As an additionally option, the dataset is available in the following github repository https://github.com/OpenNeuroDatasets/ds004504, which contains all the EEG recordings and you can use it to review the content or even can be clone but it will be required to clone it to the main directory of this project.

## 4. Methodological Contribution
Because the data used has already been collected and pre-processed, the work done in this project has been divided into the following steps: 

### EEG Data Exploration
In order to know the data, and to become familiar with the use of the different functions and methods offered by MNE for the processing of EEG data, some explorations have been made that allowed to design the next steps in the work, such as the use of Epochs creation (windows), extracting features using PSD and recognizing the behavior of signals with either the original or pre-processed data.

### Feature Extraction
During the extraction of the characteristics, it was necessary to balance the classes to avoid overfitting in the results of the predictions, likewise, the data was first converted from Raw Data to Windows data using MNE, each window has a size of 60 seconds, and an overlapping of 20 seconds, and for all participants to have the same number of windows a crop of 480 seconds was performed on all participants.

A downsampling was performed to work on the training of the models directly with the results generated in this step.

### Modeling
During the training, 3 different approaches were followed based on the data extracted during Feature Extraction. The first of them uses all the features obtained, then uses only the features of Total_Power and Standard_Deviation manually reducing the features, and finally uses the features of the frequency bands Theta and Alpha, which were previously used in other work as they are related to the behavior of the pole-front EEG signals.

The training was first carried out by performing cross-validation with 3 different methods (KFold, StratifiedKFold, and StratifiedShuffleSplit) to determine which of these offered better performance, In addition, the training was performed using 5 different ML models such as SVM, Random Forest, XGBoost, LightGBM and AdaBoost which were chosen for their performance advantages, versatility and operation with data containing a high number of features. In addition, the SelectKBest has 3 different functions for the selection of features with which to obtain better results.

After completing the Cross-Validation, and with the result of the models that presented the best results, we proceeded to make an Optimization of Hyperparameters and finally obtained the results that allowed the comparison of the predictions under the use of all the channels or only the front pole channels.

### Results
To discuss the results, 3 different performance metrics were used: Accuracy, F1 Score, and ROC AUC, in addition to confusion matrix support.

The final idea of the project has been to verify that the performance of the predictions made with the use of EEG front channels such as FP1 is reliable and that obtaining signals through BCI devices that provide a lower number of channels is feasible to develop multiple approaches to research.

## 5. Results

Using techniques such as cross-validation, feature selection, and five different classification models, we have generated predictions. Based on these predictions, we created a dashboard that compares the results of using only the FP1 channel versus using other available channels.

It can be inferred that the most satisfactory outcomes are obtained when utilizing features from all channels. Nevertheless, in some inspections, accuracies of up to +20% were achieved by using features solely from the FP1 channel. Although models still require optimization and enhancement, this result provides a certain level of confidence in relying solely on the front-end channels.

Tableau Public: [EEG Prediction Analysis from Alzheimer and Frontotemporal Dementia](https://public.tableau.com/app/profile/sobiedd.chavez/viz/EEGPredictionAnalysisfromAlzheimerandFrontotemporalDementia/Summary#1)