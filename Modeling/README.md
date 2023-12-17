# Modeling

**Author**:         Elmo Chavez\
**Date**:           October 19, 2023

**Description**:
This section contains the classification modeling by using the feature extraction datasets made in the previous step.\

1. The first work was to ensure finding the best classifier model through different cross-validation methods in combination with some feature selection methods.
   1. Preselecting only the Total Power and Spectral Density from the previous Training dataset a new cross validation approach has been performed.
   2. Preselecting only the features from the Alpha frequency band from the initial Training dataset another cross validation approach has been performed.
2. With the feature extraction similar to the previous one and omiting the usage of frequency bands, another cross validation approach has been performed.
3. Hyperparameters tunning for the XGBoost model by using the first feature extraction, and getting the feature importance to understand the prediction and the performance.
4. Another hyperparameter tunning for the Random Forest and understand the predictions and the results.

A brief summary of the findings has been posted in the main [Readme.md](https://github.com/sobieddch90/mcd-udg-tfm-eeg-classification/blob/main/README.md) file of this project.