

from utils import *

train_with_pytorch_lightning = True

s2s_args = ArgumentsS2S(
    batch_size= 16,
    max_length=512
)

path_train = "C:/Users/Admin/Downloads/ELI5.jsonl"
path_val = "C:/Users/Admin/Downloads/ELI5_val.jsonl"

dataset_train = load_dataset('json', data_files = path_train)
dataset_val = load_dataset('json', data_files = path_val)

train = dataset_train['train']
val = dataset_val['train']

train1 = train.map(lambda ex: preprocess(ex,num_docs=5), remove_columns = ['question_id'])
val1 = val.map(lambda ex: preprocess(ex,num_docs=5), remove_columns = ['question_id'])

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
    checkpoint_callback = L.pytorch.callbacks.ModelCheckpoint(
        dirpath='checkpoints',
        filename='best-checkpoint',
        save_top_k=1, verbose=True, monitor='val_loss', mode="min"
    )
    my_model = model()
    trainer = L.Trainer(accelerator='gpu', devices=1, max_epochs=3, callbacks=[checkpoint_callback], enable_progress_bar=True)
    trainer.fit(my_model, train_1, val_1)
