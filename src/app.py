import flask
import config
import torch

from flask import Flask
from flask import request
from model import BERTBasedUncased

app = Flask(__name__)

DEVICE = 'cuda'
model = None

def sentence_pred(sentence, model):
    tokenizer = config.TOKENIZER
    max_len = config.MAX_LENGTH
    review = str(sentence)
    review = " ".join(review.split())

    inputs = tokenizer.encode_plus(
        review,
        None, 
        add_special_tokens=True,
        max_length = max_len,
        truncation=True)
    
    ids = inputs["input_ids"]
    mask = inputs["attention_mask"]
    token_type_ids = inputs["token_type_ids"]

    padding_len = max_len - len(ids)
    ids = ids + ([0]*padding_len)
    mask = mask + ([0]*padding_len)
    token_type_ids = token_type_ids + ([0]*padding_len)

    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)

    ids = ids.to(DEVICE, dtype=torch.long)
    token_type_ids = token_type_ids.to(DEVICE, dtype=torch.long)
    mask = mask.to(DEVICE, dtype=torch.long)

    outputs = model(
            ids=ids, 
            att_mask=mask, 
            token_type_ids = token_type_ids
        )
    
    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]

@app.route('/predict')
def predict():
    sentence = flask.request.args.get("sentence")
    print(sentence)
    postive_pred = sentence_pred(sentence, model=MODEL)
    negative_pred = 1-postive_pred
    
    response = {}
    response['response'] = {"positive": str(postive_pred),
                            "negative": str(negative_pred),
                            "sentence": str(sentence)}
    return flask.jsonify(response)


if __name__ == "__main__":
    MODEL = BERTBasedUncased()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run()