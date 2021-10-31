## Neural_Network_Charity_Analysis

Module 19 Neural Networks Machine Learning

Completed by Angela Kumar

### Purpose

Create a deep-learning neural network to analyze and classify the success of the charitable donations.

### Overview

Explore and implement neural networks using the TensorFlow platform in Python. We'll discuss the background and history of computational neurons as well as current implementations of neural networks as they apply to deep learning. We'll discuss the major costs and benefits of different neural networks and compare these costs to traditional machine learning classification and regression models. Additionally, we'll practice implementing neural networks and deep neural networks across a number of different datasets, including image, natural language, and numerical datasets. Finally, we'll learn how to store and retrieve trained models for more robust uses.

### Resources

Data: charity_data.csv; AlphabetSoupCharity_starter_code.ipynb converted to AlphabetSoupCharity.ipynb;

Technologies: Python; Pandas; Google Colabortory; TensorFlow library; sklearn libraries;VSCode

### Background

Bek’s come a long way since her first day at that boot camp five years ago—and since earlier this week, when she started learning about neural networks! Now, she is finally ready to put her skills to work to help the foundation predict where to make investments.

With your knowledge of machine learning and neural networks, you’ll use the features in the provided dataset to help Beks create a binary classifier that is capable of predicting whether applicants will be successful if funded by Alphabet Soup.

From Alphabet Soup’s business team, Beks received a CSV containing more than 34,000 organizations that have received funding from Alphabet Soup over the years. Within this dataset are a number of columns that capture metadata about each organization, such as the following:

* EIN and NAME—Identification columns
* APPLICATION_TYPE—Alphabet Soup application type
* AFFILIATION—Affiliated sector of industry
* CLASSIFICATION—Government organization classification
* USE_CASE—Use case for funding
* ORGANIZATION—Organization type
* STATUS—Active status
* INCOME_AMT—Income classification
* SPECIAL_CONSIDERATIONS—Special consideration for application
* ASK_AMT—Funding amount requested
* IS_SUCCESSFUL—Was the money used effectively

### Deliverables

#### Deliverable 1: Preprocessing Data for a Neural Network Model

Using your knowledge of Pandas and the Scikit-Learn’s StandardScaler(), you’ll need to preprocess the dataset in order to compile, train, and evaluate the neural network model later in Deliverable 2.

The EIN and NAME columns have been dropped 
The columns with more than 10 unique values have been grouped together 
The categorical variables have been encoded using one-hot encoding 
The preprocessed data is split into features and target arrays 
The preprocessed data is split into training and testing datasets 
The numerical values have been standardized using the StandardScaler() module

#### Deliverable 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

The neural network model using Tensorflow Keras contains working code that performs the following steps:
The number of layers, the number of neurons per layer, and activation function are defined 
An output layer with an activation function is created 
There is an output for the structure of the model 
There is an output of the model’s loss and accuracy 
The model's weights are saved every 5 epochs 
The results are saved to an HDF5 file

#### Deliverable 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.

Create a new Jupyter Notebook file and name it AlphabetSoupCharity_Optimzation.ipynb.
Import your dependencies, and read in the charity_data.csv to a Pandas DataFrame.
Preprocess the dataset like you did in Deliverable 1, taking into account any modifications to optimize the model.
Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.
Create a callback that saves the model's weights every 5 epochs.
Save and export your results to an HDF5 file, and name it AlphabetSoupCharity_Optimization.h5

#### Deliverable 4: A Written Report on the Neural Network Model (README.md)

### Summary
