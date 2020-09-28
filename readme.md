# Analyzing the Probability of a Sale from Anonymized Sales Data with a Deep FF Neural Network

## Introduction

This repository examines an machine learning approach to predicting the possibility of a sale based upon anonyomized and labeled sales data metrics.

## The Data

The data consists of a collection of 160,000 anonymized records labeled with a binary 1/0 if a sale was made or not. The data consisted variously of 47 columns of float data, and 3 columns of categorical data. After eliminating rows with missing data we were left with 159,913 rows of viable data.  In order to improve convergence of our models we also normalized the data for each feature.  This is of note because during prediction, incoming data will need to be normalized to fit the model's expectations.

We encoded the categorical columns with the one hot encoding technique. One hot encoding eliminates the possibility of introducing level bias effects by limiting the identifying factor to a maximum value of 1 when present, and zero otherwise. However, adding one-hot- encoding introduces its own set of problems considering that the dataset becomes very sparse with increasing numbers of encoded features.  Sparse datasets struggle to converge.  As such we tested our models with and without these encoded features to gauge the relative performance of leaving these features in the compiled model or removing them.  

The categorical features are related to time and location (month, weekday, and continent).  The month and weekday columns might not only appropriate in a time series modeling approach.  This insight led us to attempt to train a seperate model, without these features, to see if the model can make an accurate predict a sale.

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

