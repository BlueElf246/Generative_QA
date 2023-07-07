import nlp
import pandas as pd
from utils import *
from datasets import load_dataset
#load dataset
device = 'cuda' if torch.cuda.is_available() else 'cpu'
path_train = "/Users/datle/Downloads/ELI5.jsonl"
path_val = "/Users/datle/Downloads/ELI5_val.jsonl"
dataset_train = load_dataset('json', data_files=path_train)
dataset_val = load_dataset('json', data_files=path_val)
train = dataset_train['train']
val = dataset_val['train']
# Load tokenizer and model
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)

#Argument
class ArgumentsS2S():
    def __init__(self):
        self.batch_size = 3
        self.backward_freq = 16 # perform update after 16 batch
        self.max_length = 512 # maximum token numbers for inputs
        self.print_freq = 100 # print cost, time taken after 100 batch
        self.model_save_name = "seq2seq_models/eli5_bart_model"
        self.learning_rate = 2e-4
        self.num_epochs = 3
        self.device = device

s2s_args = ArgumentsS2S()
# convert dataset to DataSet Class
train = eli5dataset(train, num_docs=1)
val = eli5dataset(val, num_docs=1)

# train
train_qa_s2s(model, tokenizer, train, val, s2s_args)

# evaluate trained model
predicted_all = []
answers_all = []
for idx in range(len(val)):
    inputs, answer = val.make_example(idx)
    predicted = qa_s2s_generate(inputs, model, tokenizer)
    predicted_all += predicted
    answers_all += answer
nlp_rouge = nlp.load_metric('rouge')

scores = nlp_rouge.compute(
    predicted_all, answers_all,
    rouge_types=['rouge1', 'rouge2', 'rougeL', 'rougeLsum'],
    use_agregator=True, use_stemmer=False
)
df = pd.DataFrame({
    'rouge1': [scores['rouge1'].mid.precision, scores['rouge1'].mid.recall, scores['rouge1'].mid.fmeasure],
    'rouge2': [scores['rouge2'].mid.precision, scores['rouge2'].mid.recall, scores['rouge2'].mid.fmeasure],
    'rougeL': [scores['rougeL'].mid.precision, scores['rougeL'].mid.recall, scores['rougeL'].mid.fmeasure],
}, index=[ 'P', 'R', 'F'])
df.style.format({'rouge1': "{:.4f}", 'rouge2': "{:.4f}", 'rougeL': "{:.4f}"})

