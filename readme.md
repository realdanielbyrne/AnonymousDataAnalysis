# Anonymous Data Classification Machine Learning Model

## Preface

Engles Family Enterprises inc., client, and  Byrne's Brains, the developer, have partnered on the Pathways cloud machine learning development project, pursuant to Statement of Work number 29155, to develop a new product setup application which will feed the client's customer metrics into a cloud hosted machine learning model to predict a customer viability metric.

## Introduction

Byrne's Brains is designing a cloud hosted machine learning model with a customized HTTP REST which will accept and classify anonymized data in a highly parallelized environment, from Engles family agents in the field.  The Cloud Interface Unit will be a compliant REST server designed to classify a collection of customer data points to give instant feedback to Engles Family agents to help in determining the viability of a sales decision. 

This cloud hosted Machine Learning model will enable quick decision making abilities by instantly quantifying a collection of diverse metrics gathered by the client and then classifying this quantity into a prediction which will then be fed back over the API to the client's calling computer.

This inaugural "Request For Comment", RFC, document  is being issued as a first step attempt to stimulate a dialog on  the emerging design of this cloud hosted machine learning model.  

This paper will go over the initial design of the machine learning model, challenges that emerged in the development process, and proposals for hosting.

## The Data

The client has generously provided the developer a collection of 160,000 anonymized records which has been used in developing an initial classification model. The data consisted variously of 46 7 columns of float data, and 3 columns of categorical data. After eliminating rows with missing data we were left with 159,913 rows of data.  

Each row was pre-labeled with the proper classification when we received the data. In order to improve convergence of our models we also normalized the data for each feature.  This is of note because during prediction, incoming data will need to be normalized to fit the model's expectations.

We encoded the categorical columns with the one hot encoding technique. One hot encoding eliminates the possibility of introducing level bias effects by limiting the identifying factor to a maximum value of 1 when present, and zero otherwise. However, adding one-hot- encoding introduces its own set of problems considering that the dataset becomes very sparse with increasing numbers of encoded features.  Sparse datasets struggle to converge.  As such we tested our models with and without these encoded features to gauge the relative performance of leaving these features in the compiled model or removing them.  The categorical features in particular seemed to be related to time and location (month,weekday, and continent).  The month and weekday columns might not only appropriate in a time series modeling approach.  This insight led us to attempt to train our models, which  are not time series models, without these features.

## The Model

Because of the large number of features, we settled on a deep neural network, DNN, architecture as our model of choice.  Deep neural networks provide the ability to encode and reduce a large numbers of feature combinations into a finite set of classification labels with relative ease.  



The model consisted of 4 fully connected layers of 300, 250,200, and 150 nodes.  The output layer is a single logistic regression node to squash the prediction to a 1 or a 0 class label.  We used rely activations on every layer except the output where we use the logistic.  Dropout was used only on the last layer.  We trained the model using gradient decent with ADAM optimizations.

## Training

Training on average took about 10 minutes on a MacBook Pro equipped with a 2.3 GHz 8-Core Intel Core i9, 32 GB 2400 MHz DDR4, and training took place on the built int Radeon Pro Vega 16 4 GB using the PlaidML Keras API.

## Results

We trained our model on the 3 filtered datasets below.  Results from validation are listed with each model.

    nofactors.csv
    Test loss: 0.25880000801213704
    Test accuracy: 0.9369949494949495

    cleanedonehot.csv
    Test loss: 0.5000135238940894
    Test accuracy: 0.884280303030303

    cleandeddata.csv
    Test loss: 0.33758714681872254
    Test accuracy: 0.9227272727272727

## Hosting

We recommend hosting the model and API on Azure Functions.  Azure functions have the benefit of configuring the service so that the client only pays when the service is actively being used. Otherwise the service goes dormant.  Furthermore , since Azure functions can store their models in long term, low cost storage buckets hosting for the model and API will be relatively inexpensive.

