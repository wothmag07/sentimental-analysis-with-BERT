from sklearn.metrics import ConfusionMatrixDisplay, pair_confusion_matrix, precision_score, accuracy_score, recall_score, f1_score
from tqdm import tqdm
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


def loss_fn(outputs, targets):
    return nn.BCEWithLogitsLoss()(outputs, targets.view(-1,1))

def train_fn(dataloader, model, optimizer, device, scheduler):
    train_loss = 0
    model.train()

    for batch_index, dataset in tqdm(enumerate(dataloader), total=len(dataloader)):
        ids = dataset['ids']
        token_type_ids = dataset['token_type_ids']
        mask = dataset['mask']
        targets=  dataset['targets']

        ids = ids.to(device, dtype=torch.long)
        token_type_ids = token_type_ids.to(device, dtype=torch.long)
        mask = mask.to(device, dtype=torch.long)
        targets = targets.to(device, dtype=torch.float)

        optimizer.zero_grad()
        #torch.cuda.empty_cache()

        outputs = model(
            ids=ids, 
            att_mask=mask, 
            token_type_ids = token_type_ids
        )
        loss = loss_fn(outputs, targets)
        train_loss += loss
        loss.backward()

        optimizer.zero_grad()

        optimizer.step()
        
        scheduler.step()
    train_loss /= len(dataloader)

    return train_loss

def evaluate_fn(dataloader, model, device):
    fin_targets = []
    fin_outputs=[]
    eval_loss = 0
    model.eval()
    
    with torch.inference_mode():
        for batch_index, dataset in tqdm(enumerate(dataloader), total=len(dataloader)):
            ids = dataset['ids']
            token_type_ids = dataset['token_type_ids']
            mask = dataset['mask']
            targets=  dataset['targets']

            ids = ids.to(device, dtype=torch.long)
            token_type_ids = token_type_ids.to(device, dtype=torch.long)
            mask = mask.to(device, dtype=torch.long)
            targets = targets.to(device, dtype=torch.float)

            outputs = model(
                ids=ids, 
                att_mask=mask, 
                token_type_ids = token_type_ids
            )

            loss = loss_fn(outputs, targets)
            eval_loss += loss

            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
        
        eval_loss /= len(dataloader)

    return fin_outputs, fin_targets, eval_loss

def loss_curve(training_loss,eval_loss, epoch_list):
    plt.plot(epoch_list, training_loss, label="Training Loss")
    plt.plot(epoch_list, eval_loss, label="Test Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()

def evaluate_model(model,data_loader,loss_fn,acc_fn, device):
    loss , acc = 0, 0
    model.eval()
    with torch.inference_mode():
        for X, Y in data_loader:
            X,Y  = X.to(torch.device), Y.type(torch.LongTensor).to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred,Y)
            acc += acc_fn(Y, y_pred.argmax(dim=1))
        loss /= len(data_loader)
        acc /= len(data_loader)
        return {'model_name':model.__class__.__name__, #this works when model was created with class
                'model_loss':loss.item(),
                'model_accuracy':float(acc)}

'''def eval_metrics(outputs, target):
    accuracy = Accuracy(task="binary", num_classes=2)
    acc = accuracy(outputs, target)
    precision_metrics = Precision(task='binary', num_classes=2, average='macro')
    P = precision_metrics(outputs, target)
    recall_metrics = Recall(task='binary', num_classes=2, average='macro')
    R = recall_metrics(outputs, target)
    f1score = F1Score(task='binary', num_classes=2)
    f1 = f1score(outputs, target)
    return acc, P, R, f1'''

def eval_metrics(targets, outputs):
    acc = accuracy_score(targets, outputs)
    P = precision_score(targets, outputs)
    R = recall_score(targets, outputs)
    f1 = f1_score(targets, outputs)
    return acc, P, R, f1

def confusion_matrix(output, targets):
    matrix = ConfusionMatrixDisplay()
    cm_tensor = matrix(output, targets)
    fig, ax = pair_confusion_matrix(targets, output)
    return fig, ax






    

