## Neural_Network_Charity_Analysis

Module 19 Neural Networks Machine Learning

Completed by Angela Kumar

### Purpose

Create a deep-learning neural network to analyze and classify the success of the charitable donations.

### Overview

Explore and implement neural networks using the TensorFlow platform in Python. We'll discuss the background and history of computational neurons as well as current implementations of neural networks as they apply to deep learning. We'll discuss the major costs and benefits of different neural networks and compare these costs to traditional machine learning classification and regression models. Additionally, we'll practice implementing neural networks and deep neural networks across a number of different datasets, including image, natural language, and numerical datasets. Finally, we'll learn how to store and retrieve trained models for more robust uses.

### Resources

Data: charity_data.csv; AlphabetSoupCharity_starter_code.ipynb converted to AlphabetSoupCharity.ipynb;

Technologies: Python; Pandas; Google Colabortory; TensorFlow library; Scikit-Learn libraries; 

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

**Original data**

<img width="400" alt="Original Charity dataset screenshot" src="https://user-images.githubusercontent.com/85860367/139569525-15b864fa-5937-4ae0-8a14-8fdc0bdb52f0.PNG">

**The EIN and NAME columns have been dropped**

<img width="400" alt="Drop column screenshot" src="https://user-images.githubusercontent.com/85860367/139569403-87f8fb7b-d587-499b-9bbf-a783516b5e91.PNG">

**The columns with more than 10 unique values have been grouped together**

<img width="350" alt="Unique charity values" src="https://user-images.githubusercontent.com/85860367/139569427-3ead74a6-cdef-45fe-9204-3cf91c483259.PNG">

**The categorical variables have been encoded using one-hot encoding**

<img width="430" alt="Application types" src="https://user-images.githubusercontent.com/85860367/139570785-b3fec82f-87da-4583-93b5-f629cf5d42a3.PNG">

<img width="430" alt="Replace app type screenshot" src="https://user-images.githubusercontent.com/85860367/139570994-9248aad2-094f-42c8-85c3-3054a3905714.PNG">

<img width="430" alt="Classification screenshot" src="https://user-images.githubusercontent.com/85860367/139570822-f4d08a58-a314-481f-b08a-63799624c49e.PNG">

<img width="430" alt="Replace cls class screenshot" src="https://user-images.githubusercontent.com/85860367/139570968-9f84ca2b-b50e-4cc2-8c80-852faee76841.PNG">

<img width="430" alt="Onehotencoder screenshot" src="https://user-images.githubusercontent.com/85860367/139569494-a2a6af9f-6b14-4290-b15c-a62704d2ba6a.PNG">


**The preprocessed data is split into features and target arrays**
**The preprocessed data is split into training and testing datasets**
**The numerical values have been standardized using the StandardScaler() module**

<img width="430" alt="Split and Scale" src="https://user-images.githubusercontent.com/85860367/139569621-fa7f1f07-a193-438e-aae2-adb91614e4d4.PNG">

**Merged DataFrame**

<img width="430" alt="Merge screenshot" src="https://user-images.githubusercontent.com/85860367/139570746-9ef83860-7af3-4c4e-867f-7c982a041a3c.PNG">

#### Deliverable 2: Compile, Train, and Evaluate the Model

Using your knowledge of TensorFlow, you’ll design a neural network, or deep learning model, to create a binary classification model that can predict if an Alphabet Soup–funded organization will be successful based on the features in the dataset. You’ll need to think about how many inputs there are before determining the number of neurons and layers in your model. Once you’ve completed that step, you’ll compile, train, and evaluate your binary classification model to calculate the model’s loss and accuracy.

The neural network model using Tensorflow Keras contains working code that performs the following steps:

The number of layers, the number of neurons per layer, and activation function are defined

An output layer with an activation function is created 

There is an output for the structure of the model 

There is an output of the model’s loss and accuracy 

The model's weights are saved every 5 epochs

The results are saved to an HDF5 file

<img width="430" alt="Defining model Deliverable 2" src="https://user-images.githubusercontent.com/85860367/139571218-b8f025df-e870-46d5-8300-4a5d27dc5aee.PNG">

<img width="451" alt="Compile and Train Deliverable 2" src="https://user-images.githubusercontent.com/85860367/139571331-fc887052-fdf4-4e7c-9f78-0c9a3ae7e97e.PNG">

<img width="427" alt="Model evaluation Deliverable 2" src="https://user-images.githubusercontent.com/85860367/139571421-d67f03bc-7497-4ccc-acce-a0c15ebe43f9.PNG">

<img width="430" alt="Evaluation h5 with 50 epochs" src="https://user-images.githubusercontent.com/85860367/139571438-35ee9bee-21ee-48de-9c07-b79b6c563ac1.PNG">

<img width="583" alt="Evaluation h5 with 100 epochs" src="https://user-images.githubusercontent.com/85860367/139571445-1d4737c0-f2f4-4ed4-a2fd-f3627cf111ab.PNG">

#### Deliverable 3: Optimize the Model

Using your knowledge of TensorFlow, optimize your model in order to achieve a target predictive accuracy higher than 75%. If you can't achieve an accuracy higher than 75%, you'll need to make at least three attempts to do so.

Create a new Jupyter Notebook file and name it AlphabetSoupCharity_Optimzation.ipynb.

Import your dependencies, and read in the charity_data.csv to a Pandas DataFrame.

Preprocess the dataset like you did in Deliverable 1, taking into account any modifications to optimize the model.

Design a neural network model, taking into account any modifications that will optimize the model to achieve higher than 75% accuracy.

Create a callback that saves the model's weights every 5 epochs.

Save and export your results to an HDF5 file, and name it AlphabetSoupCharity_Optimization.h5

<img width="420" alt="Deliverable 3 logistic model" src="https://user-images.githubusercontent.com/85860367/139574489-1094e95b-7000-4dde-a78f-bda4d025db92.PNG">

<img width="420" alt="Output of Logistic model" src="https://user-images.githubusercontent.com/85860367/139574501-36d75212-c720-4659-9c78-c8d881307dbe.PNG">

<img width="420" alt="Deliverable 3 SVM" src="https://user-images.githubusercontent.com/85860367/139574515-4bda39a0-d96b-4973-9797-0aedeadc184e.PNG">

<img width="420" alt="Deliverable 3 Deep Neural Net" src="https://user-images.githubusercontent.com/85860367/139574528-b57ee687-8571-4947-9f98-41579bc8d1b1.PNG">

<img width="420" alt="deliverable 3 deep neural output" src="https://user-images.githubusercontent.com/85860367/139574548-071eb42e-8fcc-4a44-88ff-419c640098dd.PNG">

#### Deliverable 4: A Written Report on the Neural Network Model (README.md)

*Data Preprocessing

* What variable(s) are considered the target(s) for your model?

The target for the model is the IS_SUCCESSFUL column which states whether the investment was worthwhile.

* What variable(s) are considered to be the features for your model?

All other column names are considered to be the features for the model
    APPLICATION_TYPE—Alphabet Soup application type
    AFFILIATION—Affiliated sector of industry
    CLASSIFICATION—Government organization classification
    USE_CASE—Use case for funding
    ORGANIZATION—Organization type
    STATUS—Active status
    INCOME_AMT—Income classification
    SPECIAL_CONSIDERATIONS—Special consideration for application
    ASK_AMT—Funding amount requested

* What variable(s) are neither targets nor features, and should be removed from the input data?

EIN and NAME—Identification columns were removed from the data as these were neither targets nor features

*Compiling, Training, and Evaluating the Model

* How many neurons, layers, and activation functions did you select for your neural network model, and why?

<img width="500" alt="Summary of Neural Network" src="https://user-images.githubusercontent.com/85860367/139573768-b4fcf917-ac83-4aab-a6c3-fea1ce5a6d9a.PNG">

The reason to change the neurons, layers, and activation functions was to optimize the model and increase the predictive accuracy to over 75%. 

* Were you able to achieve the target model performance?

The target model performance of 75% or greater was not met. The model performance was above 72.5% in all of the optimization.  

* What steps did you take to try and increase model performance?

I selected a few things first I had binned the application type with a larger number, as well as the classification; however doing so, the performance was so much more worse with loss greater than 60% and the accuracy dropped to ~50%.  I ended up switching the numbers back and re-ran the model from the start.

Then I attemped to run the Support Vector Machine (SVM) Model and stopped the process as it was taking too long to get the results.  This was too large of a dataset for the SVM Model.

Tried the logistic model and the deep neural net model to increase performance accuracy.

I finally increased the first layer neurons to 100, changed the second layer activation to "tanh", added a third layer, and kept all other variables the same.  The result achieved was relatively minimal from .7277 to .7265

### Summary

In summary, the dataset was rather large, it is possible that this needed to be cleaned further, to avoid possible duplicates and also to filter the data to smaller groups.  The smaller grouping of data may increase the performance accuracy; however there still would be other factors to take into consideration such as the number of hidden layers, epochs and hidden activation.

A recommendation would be to consider supervised machine learning techniques to obtain a better accuracy score and internal controls.  There must be analysis and observance conducted by a human to ensure that the data is clean and free from error, fraud, waste and abuse.  There will need to be frequent discussion between the analysts, Finance, and IT.

Nonprofits are especially vulnerable to fraud compared with the for-profit sector. These include a lack of resources, poor internal controls, inadequate training, high turnover with low employee investment, poor technological resources and other factors.

Cybersecurity is just one aspect of the fraud problem among nonprofits. Vendor fraud is another. So are internal malfeasance by in-house employees — including fraudulent financial statements, embezzlement and misappropriation of assets. “Cash skimming” is one of the likeliest forms of fraud in the nonprofit sector, since “misplacing” it is all too easy to do.

Nonprofits often don’t have the resources to invest in personal and professional development programs. That means employees may not be as vigilant as they should be. This lack of personal investment can result directly in malfeasance, including diverted contributions, “phantom” vendors, compensation fraud and more

https://www.nonprofitpro.com/post/why-nonprofits-are-more-vulnerable-to-fraud-than-for-profit-businesses/

