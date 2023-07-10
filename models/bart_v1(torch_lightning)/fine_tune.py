from utils import *
path_train = "/Users/datle/Downloads/ELI5.jsonl"
path_val = "/Users/datle/Downloads/ELI5_val.jsonl"
dataset_train = load_dataset('json', data_files=path_train)
dataset_val = load_dataset('json', data_files=path_val)
train = dataset_train['train']
val = dataset_val['train']

class ArgumentsS2S():
    def __init__(self):
        self.batch_size = 1
        self.max_length = 128

s2s_args = ArgumentsS2S()

tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
train1 = eli5dataset(train, num_docs =2)
val1 = eli5dataset(val, num_docs =2)
train_1 = data_loader(train1, s2s_args, tokenizer)
val_1 = data_loader(val1, s2s_args, tokenizer)
my_model = bart_model()
trainer = L.Trainer(max_epochs=3)  # accelerator='gpu', devices=2,
trainer.fit(my_model, train_1, val_1)