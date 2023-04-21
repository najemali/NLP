###########################################################################
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch

import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
###########################################################################


class model(nn.Module):

    def __init__(self, Roberta,device):
        super().__init__()

        self.roberta = Roberta.to(device)
        self.l1 = nn.Linear(in_features=768, out_features=768)
        #self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=0.1)
        self.l2 = nn.Linear(in_features=768, out_features=1536)
        self.l3 = nn.Linear(in_features=1536, out_features=3)

    def forward(self, x, attention_mask):

        x = self.roberta(x,attention_mask)
        x = x.pooler_output
        x = self.l1(x)
        #x = self.relu(x)
        x = self.drop(x)
        x = self.l2(x)
        x = self.drop(x)
        x = self.l3(x)

        return x


def train(model, loss_fcn, device, optimizer, max_epochs, train_dataloader, val_dataloader):

    epoch_list = []
    scores_list = []
    lowest_loss = 1

    # loop over epochs
    for epoch in range(max_epochs):
        model.train()
        losses = []
        # loop over batches
        for i, data in enumerate(train_dataloader):
            optimizer.zero_grad()
            inputs, mask, labels = data
            outputs = model(inputs.to(device),mask.to(device))
            # compute the loss
            loss = loss_fcn(outputs, labels.to(device))
            # optimizer step
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        loss_data = np.array(losses).mean()
        print("Epoch {:05d} | Loss: {:.4f}".format(epoch, loss_data))
        
        if epoch % 5 == 0:
            # evaluate the model on the validation set
            # computes the f1-score
            score_list_batch = []

            model.eval()
            with torch.no_grad():
                for i, batch in enumerate(val_dataloader):
                    inputs, mask, labels = batch
                    output = model(inputs.to(device),mask.to(device))
                    loss_test = loss_fcn(output, labels.to(device))
                    predict = torch.argmax(output,axis=1)
                    score = accuracy_score(labels.cpu().numpy(), predict.cpu().numpy())
                    score_list_batch.append(score)

            score = np.array(score_list_batch).mean()
            print("Accuracy-Score: {:.4f}".format(score))
            scores_list.append(score)
            epoch_list.append(epoch)

    return epoch_list, scores_list, model