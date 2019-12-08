# Preface

Engles Family Enterprises inc., client, and  Byrne's Brains, the developer, have partnered on the Pathways cloud machine learning development project, pursuant to Statement of Work number 29155, to develop a new product setup application which will feed the client's customer metrics into a cloud hosted machine learning model to predict a customer viability metric.

# Introduction

Byrne's Brains is designing a cloud hosted machine learning model with a customized HTTP REST which will accept and classify anonyomized data in a highly parallized environment, from Engles family agents in the field.  The Cloud Interface Unit will be a compliant REST server designed to classify a collection of customer datappoints to give instant feedback to Engles Family agents to help in determinign the viability of a sales decision. 

This cloud hosted Machine Learning model will enable quick decision making abilities by instantly quantifying a collection of diverse metrics gathered by the client and then classifying this quantity into a prediction which will then be fed back over the API to the client's calling computer.

This inaugural "Request For Comment", RFC, document  is being issued as a first step attempt to stimulate a dialog on  the emerging desgin of this cloud hosted machine learning model.  

This paper will go over the initial design of the machine learning model, challenges that emerged in the development process, and proposals for hosting.

# The Data

The client has generously provided the devleloper a collection of 160,000 anonoymized records which has been used in developing an initial classification model. The data consisted variously of 46 7 columns of float data, and 3 columns of categorical data. After eliminating rows with missing data we were left with 159,913 rows of data.  

Each row was pre-labeled with the proper classification when we received the data. In order to improve convergence of our models we also normalized the data for each feature.  This is of note because during prediction, incoming data will need to be normalized to fit the model's expectations.

We encoded the categorical columns with the one hot encoding technique. One hot encoding eliminates the posibility of introducing level bias effects by limiting the identifying factor to a maximum value of 1 when present, and zero otherwise. However, adding one-hot- encoding introduces its own set of problems considering that the dataset becomes very sparse with increasing numbers of encoded features.  Sparse datasets struggle to converge.  As such we tested our models with and without these encoded features to gauge the relative performance of leaving these features in the compiled model or removing them.  The categorical features in particular seemed to be related to time and location (month,weekday, and continent).  The motnh and weekday columns might not only appropriate in a time series modeling approach.  This insight led us to attempt to train our models, which  are not time series models, without these features.
