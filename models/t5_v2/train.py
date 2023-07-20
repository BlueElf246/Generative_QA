import torch

from utils import *
from utils1 import *

prepared=False

if prepared==False:

    path_train = "C:/Users/Admin/Downloads/ELI5.jsonl"
    path_val = "C:/Users/Admin/Downloads/ELI5_val.jsonl"

    dataset_train = load_dataset('json', data_files = path_train)
    dataset_val = load_dataset('json', data_files = path_val)

    train = dataset_train['train'].select(range(1,100))
    val = dataset_val['train'].select(range(1,100))

    train1 = train.map(lambda ex: preprocess(ex,num_docs=5), remove_columns = ['question_id'])
    val1 = val.map(lambda ex: preprocess(ex,num_docs=5), remove_columns = ['question_id'])

    train_add_sep = train1.map(add_sep_token)
    val_add_sep = val1.map(add_sep_token)

    train_add_eos = train_add_sep.map(add_eos_token)
    val_add_eos = val_add_sep.map(add_eos_token)

    train_tokenized = train_add_eos.map(convert_to_features, batched=True, remove_columns=['question','answers','ctxs'])
    val_tokenized = val_add_eos.map(convert_to_features, batched=True, remove_columns=['question','answers','ctxs'])

    columns = ['input_ids', 'decoder_input_ids', 'attention_mask', 'decoder_attention_mask']
    train_tokenized.set_format(type='torch', columns=columns)
    val_tokenized.set_format(type='torch', columns=columns)

    torch.save(train_tokenized, 'train_data.pt')
    torch.save(val_tokenized, 'valid_data.pt')
else:
    train_tokenized = torch.load("train_data.pt")
    val_tokenized = torch.load("valid_data.pt")

training_args = TrainingArguments(output_dir="model/",
                                  per_device_train_batch_size=4,
                                  per_device_eval_batch_size=4,
                                  gradient_accumulation_steps=16,
                                  learning_rate=1e-4,
                                  num_train_epochs=7,
                                  logging_steps=100,
                                  run_name="QA_question_answering",
                                  evaluation_strategy="steps",
                                  save_steps=500,
                                  optim='adamw_torch'
                                  )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_tokenized,
    eval_dataset= val_tokenized,
    data_collator=T2TDataCollator()
)

# Training
trainer.train()