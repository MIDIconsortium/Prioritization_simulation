Code to reproduce the brain abnormality simulation study in "Automated triaging of head MRI scans using convolutional neural networks" (Wood et al. 2021)

In order to use this script, a three-column .csv file is needed, containing 'Accession number' (a unique identifier for each MRI head exam), 'Event Date' (date of MRI head examination),
and 'Date Last Ver' (the date that the radiology report for this examination was finalized).

A dictionary {'Accession number':pobability} is needed, mapping accession numbers to the predictions of a computer vision model (e.g., convolutional neural network) 
for this exam. The probability represents the likelihood that the head MRI contains an abnormality and is used to prioritize the reporting of this examination.

A second dictionary {'Accession number':ground_truth_probability} is needed, which maps accession numbers to the true label for this exam (in this case derived from the radiology report
using a text classification model). This is used to stratify the report delay by class (i.e., 'normal' and 'abnormal')
