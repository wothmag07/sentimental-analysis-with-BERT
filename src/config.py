import transformers

MAX_LENGTH = 512
TRAIN_BATCH_SIZE = 16
VALID_BATCH_SIZE = 8
TRAINING_EPOCHS = 10
BERT_PATH = r"D:\Workspace\NLP project\sentimental-analysis\input\bert-base-uncased"
MODEL_PATH = r"D:\Workspace\NLP project\sentimental-analysis\model.bin"
INPUT_FILE = r"D:\Workspace\NLP project\sentimental-analysis\input\dataset\IMDB Dataset.csv"
TOKENIZER = transformers.BertTokenizer.from_pretrained(BERT_PATH, do_lower_case=True)