# ML_project

This project aims to predict whether or not a higgs boson is created in a high speed proton collision. The data to train the model is train.csv and the data on which the model was tested is test.csv. ultimately we got an accuracy of 0.833 and an F-score of 0.746. To recreate this result, you need to:
- Install Numpy, this is the only external library that is used
- Download the test and train datasets
- Run the first two cells in run.ipynb

Below is a brief overview of how the python scripts are used, but for further details, see the report.

data_loading_and_preprocessing.py
Functions which are used to process the data, including:
- standardizing the data
- replacing missing values by the median
- removing zero variance columns 
- removing columns with low correlation to y

find_hyperparameter.py
In this file there are the functions which are used to find the best hyperparameters for a given input range. The functions make use of the functions in Augmenting.py and implementations.py

Augmenting.py contains the functions which are used to augment the data and implementations.py has all the machinelearning functions and their helpers, such as ridge regression, logistic regression and gradient descent.

Precise descriptions of the functions and their required input is given at the beginning of each function. 
If you have any questions about the code or the project, don’t hesitate to contact us via:
jurriaan.schuring@epfl.ch
henrik.myhre@epfl.ch
andrea.perozziello@epfl.ch
