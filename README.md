#Aspect Based Sentiment Analysis

## Introduction
The goal of this assignment is to implement a classifier that predicts opinion polarities (positive, negative or neutral) for given aspect terms in sentences. The classifier takes as input 3 elements: a sentence, an aspect term occurring in the sentence, and its aspect category. For each input triple, it produces a polarity label: positive, negative or neutral.

## About the Dataset
The dataset is in a tsv format, where each line represents 5 tab separated fields. the polarity of the opinion (the ground truth polarity label), the aspect category on which the opinion is expressed, a specific target term, the character offsets of the term (start:end), and the sentence in which the term occurs and the opinion is expressed.
<br> The training file contains 1503 opinions, and the development dataset is used to measure the performance of the classifier created on the training set. It has 376 opinions

## Requirements and about the environment
The model was developed on the following: <br>
a. python <=3.9.x <br>
b. pytorch = 1.13.1 <br>
c. pytorch-lightning = 1.8.1 <br>
d. transformers = 4.22.2 <br>
e. datasets = 2.9.0 (just the library ‘datasets’, no labelled data) <br>
f. sentencepiece = 0.1.97 <br>
g. scikit-learn = 1.2.0 <br>
h. numpy = 1.23.5 <br>
i. pandas = 1.5.3 <br>
j. nltk = 3.8.1 <br>
k. stanza = 1.4.2 <br>

## About the Model
This code implements a text classification model based on the Roberta model. The model is fine-tuned on a custom dataset for sentiment analysis. The code is divided into several functions and a Classifier class, which is responsible for training the model and predicting sentiment labels.

## Code Structure
class model(nn.Module): This class defines the custom model architecture, which consists of a pre-trained Roberta model followed by three linear layers.

train(): This function is responsible for training the custom model. It takes in the model, loss function, device, optimizer, number of epochs, train dataloader, and validation dataloader as input arguments.

roberta_preprocessing(): This function preprocesses the dataset, tokenizes the text, and creates input IDs and attention masks.

get_train_val_dataloader(): This function takes in the training and development dataset file paths, preprocesses the data, and creates the train and validation dataloaders.

get_test_dataloader(): This function takes in the test dataset file path, preprocesses the data, and creates the test dataloader.

class Classifier: This class contains the constructor, __init__(), and two methods, train() and predict(). The train() method trains the classifier model, and the predict() method predicts the sentiment labels for a given dataset.

## Dependencies
a. torch <br>
b. transformers <br>
c. numpy <br>
d. pandas <br>
e. scikit-learn <br>

## Results of the model
From our code, we have been able to get a mean accuracy on the test set of XX% in XX time.

## Further Improvements

1. Implementation of more layers in the model architecture to increase depth of the model
2. Model ensembling to train multiple models and combine their predictions using methods like majority voting, weighted averaging, or stacking. This can help improve the overall performance and robustness of the final model.
3. Better handling of class imbalance using oversampling, undersampling techniques like SMOTE.


