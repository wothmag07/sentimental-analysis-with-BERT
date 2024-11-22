import config
import preprocess
import torch
import torch.nn as nn
import numpy as np
from model import BERTBasedUncased
from torch.optim import AdamW
import engine
#from transformers import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, classification_report

def run():

    train_dataloader, valid_dataloader, len_train, _ = preprocess.preprocess(config.INPUT_FILE)

    device = torch.device("cuda")
    model = BERTBasedUncased()

    model.to(device)

    param_optimizers = list(model.named_parameters())
    #print(param_optimizers)
    no_decay = ['bias','LayerNorm.bias','LayerNorm.weight']
    optimizer_parameters = [
        {'params': [p for n,p in param_optimizers if not any(nd in n for nd in no_decay)], 'weight_decay':0.001},
        {'params': [p for n,p in param_optimizers if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
    ]
    #print(optimizer_parameters)

    num_train_steps = int(len_train / (config.TRAIN_BATCH_SIZE * config.TRAINING_EPOCHS))

    optimizer = AdamW(optimizer_parameters, lr = 3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_train_steps)
    
    #model = nn.DataParellel(model)

    accuracy_best = 0

    accuracy = 0

    epoch_list = []
    training_loss = []
    val_loss = []
    for epoch in range(config.TRAINING_EPOCHS):
        train_loss = engine.train_fn(train_dataloader, model=model, optimizer=optimizer, scheduler=scheduler, device=device)
        #training_loss.extend(train_loss)
        outputs, targets, eval_loss = engine.evaluate_fn(valid_dataloader, model=model, device=device)
        #.extend(eval_loss)
        outputs = np.array(outputs).reshape(-1) >= 0.5  # Flatten to (5000,)
        targets = np.array(targets).reshape(-1)  # Ensure targets is also 1D

        # print(len(outputs))  # Should be 5000
        # print(len(targets))  # Should also be 5000

        accuracy, precision, recall, f1score =  engine.eval_metrics(outputs=outputs, targets=targets)

        print("Epoch {}/{},  Accuracy: {:.3f}".format(epoch+1,config.TRAINING_EPOCHS, accuracy))
        print(f"Precision : {precision:.3f} | Recall : {recall:.3f} | F1 Score : {f1score:.3f}")
        
        if accuracy > accuracy_best:
            torch.save(model.state_dict(), config.MODEL_PATH)
            accuracy_best = accuracy

        epoch_list.append(epoch)

    #engine.loss_curve(training_loss, val_loss, epoch_list)

if __name__ == "__main__":
    run()