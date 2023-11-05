# Feature Extraction

**Author**:         Elmo Chavez\
**Date**:           October 8, 2023

**Description**:
The present directory contains the notebooks developed with the objective of getting the features for the training dataset that will be used to develop the Classifier Model. The features will be divided from the following characteristics of the data such as:
- Epochs (Windows)
- Channels (EEG)
- Frequency Bands
And finally by the usage of the PSD method to get features such as:
- Total Power
- Relative Power
- Average Power
- Peak to Peak
- Standard Deviation
- Spectral Entropy

> The previous description means that the number of features will depend on the number of each attribute previously mentioned and multiplied one with the 6 features:\
> Example:\
> 10 Epochs, 5 Channels, 5 Frequency Bands; multiplied with 6 will end in 1500 features. 

Here you can find the following notebooks:
1. **Preselect EEG Datasets**: The total subjects are divided into 3 classes, Alzheimer Disease, Frontotemporal Dementia and Healthy Control. The last one is not necessary for the model, due to the approach is supported with the MMSE method which already has the independecy to decide a subject is healthy or not. In addition, the dataset are unbalanced, to ensure a better performance of the model the number of subject for each disease should be the same, if not, an overfitting could be found.
2. **Feature Extraction PSD**: With the MNE support and the PSD method which allows to get the Power Spectral Density in a frequency range, we can obtain the list of features we want and also separate the frequency in bands. The final result of this notebook will be the two datasets for the traning separated into the two approaches (All Channels and FP1 channel).

>The training dataset can be found in the `Training Dataset` directory

Next Step after the Feature Extraction is [Modeling](https://github.com/sobieddch90/mcd-udg-tfm-eeg-classification/blob/main/Modeling)