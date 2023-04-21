from typing import List

import torch

################################## imports
import roberta_preprocessing as rp
import model as mdl

import numpy as np

from transformers import RobertaTokenizer, RobertaModel, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup

from sklearn.utils.class_weight import compute_class_weight

import torch.nn as nn
##########################################




class Classifier:
    """
    The Classifier: complete the definition of this class template by providing a constructor (i.e. the
    __init__() function) and the 2 methods train() and predict() below. Please donot change
     """

    def __init__(self):
        self.train_dataloader = None
        self.val_dataloader = None
        self.test_dataloader = None
        self.model = None


    ############################################# comp
    def train(self, train_filename: str, dev_filename: str, device: torch.device):
        """
        Trains the classifier model on the training set stored in file trainfile
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
          - DO NOT USE THE DEV DATA AS TRAINING EXAMPLES, YOU CAN USE THEM ONLY FOR THE OPTIMIZATION
         OF MODEL HYPERPARAMETERS
        """

        # Preprocess the train_filename and dev_filename to get the train and val dataloader
        self.train_dataloader, self.val_dataloader, y_train = rp.get_train_val_dataloader(train_filename,dev_filename)


        # Define the model
        Roberta = RobertaModel.from_pretrained("roberta-base").to(device)
        basic_model = mdl.model(Roberta,device).to(device)


        # Define hyperparameters
        # max number of epochs
        #max_epochs = 25
        max_epochs = 1

        # loss Function
        class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.FloatTensor(class_weights).to(device)

        loss_fcn = nn.CrossEntropyLoss(weight=class_weights)

        # optimizer
        optimizer = torch.optim.AdamW(basic_model.parameters(), lr=2e-5)
        scheduler = get_linear_schedule_with_warmup(optimizer,0,max_epochs*len(self.train_dataloader))


        # Train the model
        epoch_list, basic_model_scores, self.model = mdl.train(basic_model, loss_fcn, device, optimizer, max_epochs, self.train_dataloader, self.val_dataloader)



    def predict(self, data_filename: str, device: torch.device) -> List[str]:
        """Predicts class labels for the input instances in file 'datafile'
        Returns the list of predicted labels
        PLEASE:
          - DO NOT CHANGE THE SIGNATURE OF THIS METHOD
          - PUT THE MODEL and DATA on the specified device! Do not use another device
        """

        # Preprocess the data_filename to get the test dataloader
        self.test_dataloader = rp.get_test_dataloader(data_filename)

        # Get the model
        model = self.model


        predictions = []
        model.eval()

        for i, batch in enumerate(self.test_dataloader):
            inputs, mask, labels = batch
            output = model(inputs.to(device),mask.to(device))
            #loss_test = loss_fcn(output, labels.to(device))
            predict = torch.argmax(output,axis=1)
            predictions.extend(predict.cpu().tolist())

        def convert_labels(predictions: List[int]) -> List[str]:
          label_names = ['negative', 'neutral', 'positive']
          return [label_names[pred] for pred in predictions]
        
        converted_predictions = convert_labels(predictions)
      
        return converted_predictions
