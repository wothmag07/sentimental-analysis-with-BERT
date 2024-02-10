import torch.nn as nn
import transformers
import config

class BERTBasedUncased(nn.Module):
    def __init__(self):
        super(BERTBasedUncased, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_dropout = nn.Dropout(0.3)
        
        self.classifier = nn.Sequential(
            nn.Linear(768, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )


    def forward(self, ids, att_mask, token_type_ids):
        _, output2 = self.bert(ids, 
                               attention_mask=att_mask, 
                               token_type_ids=token_type_ids,
                               return_dict=False)
        bert_output = self.bert_dropout(output2)
        logits = self.classifier(bert_output)
        return logits


