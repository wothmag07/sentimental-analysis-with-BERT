import pandas as pd
import config
import torch
import dataset
from sklearn.model_selection import train_test_split

def preprocess(dataset_path):

    
    df = pd.read_csv(dataset_path).fillna(0)
    df.sentiment = df.sentiment.apply(lambda x: 1 if x == 'positive' else 0)

    df_train_inputs, df_valid_inputs = train_test_split(df, 
                                                        test_size=0.1, 
                                                        random_state=42, 
                                                        shuffle=True,
                                                        stratify=df.sentiment.values)
    

    df_train= df_train_inputs.reset_index(drop=True)
    df_valid= df_valid_inputs.reset_index(drop=True)

    len_train = len(df_train)
    len_valid = len(df_valid)

    train_dataset = dataset.IMDBDataset(review=df_train.review.values,
                                        target=df_train.sentiment.values)
    
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                  batch_size = config.TRAIN_BATCH_SIZE,
                                                  num_workers=4)
    
    valid_dataset = dataset.IMDBDataset(review=df_valid.review.values,
                                        target=df_valid.sentiment.values)
    
    valid_dataloader = torch.utils.data.DataLoader(valid_dataset,
                                                  batch_size = config.VALID_BATCH_SIZE,
                                                  num_workers=1)
    
    return train_dataloader, valid_dataloader, len_train, len_valid
    
    



