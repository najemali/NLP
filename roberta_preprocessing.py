###########################################################################
import pandas as pd
import numpy as np
import re, string
import nltk.data
import nltk
from matplotlib import pyplot as plt
#%matplotlib inline

# Download English 
nltk.download('stopwords')
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

pd.set_option('max_colwidth', None)

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch

from transformers import RobertaTokenizer
###########################################################################




def roberta_preprocessing(data):
    # Rename columns
    data = data.rename(columns={0: 'Sentiment', 1: 'Category', 2: 'Subject', 3: 'Index', 4: 'Text'})

    # Lowercase text and subject
    data['Text'] = data['Text'].str.lower()
    data['Subject'] = data['Subject'].str.lower()

    # Insert special token in text
    data[['Start', 'End']] = data['Index'].str.split(':', 1, expand=True)
    data['Start'] = data['Start'].astype(int)
    data['End'] = data['End'].astype(int)
    data['Text'] = data.apply(lambda x: x['Text'][:x['Start']] + '\"' + x['Text'][x['Start']:x['End']] + '\"' + x['Text'][x['End']:], axis=1)

    # Drop unnecessary columns
    data = data.drop(columns=['Index', 'Start', 'End'])

    # Split category column into main and sub categories
    data['Category'] = data['Category'].str.lower()
    data[['Main_Category', 'Sub_Category']] = data['Category'].str.split('#', 1, expand=True)
    data['Sub_Category'] = data['Sub_Category'].str.replace('_', ' ')
    data['Category'] = data['Main_Category'] + ' ' + data['Sub_Category']

    # Append category to text
    data['Text'] = data['Text'] + ' <s> ' + data['Category'] + ' </s>'

    # Label encode sentiment column
    data['Sentiment'] = data['Sentiment'].apply(lambda x: 2 if x == 'positive' else (1 if x == 'neutral' else 0))

    return data


def get_train_val_dataloader(train_filename,dev_filename):

    # preprocess the train data
    train = pd.read_csv(train_filename,sep='\t',header=None)
    train = roberta_preprocessing(train)


    # tokenize
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    train['Input'] = train['Text'].apply(lambda x: tokenizer(x, padding='max_length', max_length=100)['input_ids'])
    train['Mask'] = train['Text'].apply(lambda x: tokenizer(x, padding='max_length', max_length=100)['attention_mask'])


    # create train and val dataloader
    X = train[['Input','Mask']]
    y = np.array(train['Sentiment'].tolist())
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


    train_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(X_train['Input'].tolist()), dtype=torch.long),
                                           torch.tensor(np.array(X_train['Mask'].tolist()), dtype=torch.long),
                                           torch.tensor(y_train, dtype=torch.long))
    val_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(X_val['Input'].tolist()), dtype=torch.long),
                                         torch.tensor(np.array(X_val['Mask'].tolist()), dtype=torch.long),
                                         torch.tensor(y_val, dtype=torch.long))

    #train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True,num_workers=2) # REMOVE COMMENT IF YOU USE GPU
    #val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=32,shuffle=False,num_workers=2)

    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=64,shuffle=True) # ADD COMMENT IF YOU USE CPU
    val_dataloader = torch.utils.data.DataLoader(val_dataset,batch_size=32,shuffle=False)

    return train_dataloader, val_dataloader, y_train


def get_test_dataloader(data_filename):

    # preprocess the train data
    test = pd.read_csv(data_filename,sep='\t',header=None)
    test = roberta_preprocessing(test)

    # tokenize
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    test['Input'] = test['Text'].apply(lambda x: tokenizer(x, padding='max_length', max_length=100)['input_ids'])
    test['Mask'] = test['Text'].apply(lambda x: tokenizer(x, padding='max_length', max_length=100)['attention_mask'])

    # create test dataloader
    X_test = test[['Input','Mask']]
    y_test = np.array(test['Sentiment'].tolist())

    test_dataset = torch.utils.data.TensorDataset(torch.tensor(np.array(X_test['Input'].tolist()), dtype=torch.long),
                                          torch.tensor(np.array(X_test['Mask'].tolist()), dtype=torch.long),
                                          torch.tensor(y_test, dtype=torch.long))

    #test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=False,num_workers=2) # REMOVE COMMENT IF YOU USE GPU
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=32,shuffle=False) # ADD COMMENT IF YOU USE CPU

    return test_dataloader
