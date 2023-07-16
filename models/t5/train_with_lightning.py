import torch.cuda

from utils import *

train_with_pytorch_lightning = False

s2s_args = ArgumentsS2S(
    batch_size= 16,
    max_length=256
)

path_train = "/Users/datle/Downloads/ELI5.jsonl"
path_val = "/Users/datle/Downloads/ELI5_val.jsonl"

dataset_train = load_dataset('json', data_files = path_train)
dataset_val = load_dataset('json', data_files = path_val)

train = dataset_train['train'].select(range(1,1000))
val = dataset_val['train']

train1 = train.map(lambda ex: preprocess(ex,num_docs=3), remove_columns = ['question_id'])
val1 = val.map(lambda ex: preprocess(ex,num_docs=3), remove_columns = ['question_id'])

tokenizer = T5Tokenizer.from_pretrained("t5-base")

train2 = eli5dataset(train1)
val2 = eli5dataset(val1)

train_1 = data_loader(train2, s2s_args, tokenizer)
val_1 = data_loader(val2, s2s_args, tokenizer)

if train_with_pytorch_lightning == False:
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    model = T5ForConditionalGeneration.from_pretrained("t5-base").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4)
    fit(model, optimizer, tokenizer, train_1, val_1, epochs=3, device=device)
else:
    my_model = model()
    trainer = L.Trainer(max_epochs=3)
    #accelerator='gpu', devices=2,
    trainer.fit(my_model, train_1, val_1)
