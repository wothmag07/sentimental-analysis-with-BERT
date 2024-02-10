import transformers

MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 2
TRAINING_EPOCHS = 10
ACCUMULATION = 2
BERT_PATH = r"D:\Workspace\NLP project\bert-base-uncased"
MODEL_PATH = "model.bin"
INPUT_FILE = r"D:\Workspace\NLP project\input\IMDB Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)
