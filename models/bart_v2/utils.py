from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch
import functools
from transformers import AdamW, get_linear_schedule_with_warmup
import math
from time import time
from datasets import load_dataset
from tqdm import tqdm
def replace_text(string):
    return string.replace("\'","").replace("\n","").replace("URL_0","").lower().strip()
def preprocess(ex, num_docs=3):
    ex['question'] = replace_text(ex['question'])
#     context = [k[0] for k in ex['ctxs'][:3]]
#     context = replace_text(' '.join(context))
    context = ex['ctxs'][:num_docs]
    if type((context[0])) == list:
        context = [k[0] for k in context]
    context = replace_text(' '.join(context))
    ex['ctxs'] = context
    ex['answers'] = [replace_text(i) for i in ex['answers']]
    return ex

class eli5dataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.qa_id_list = [
            (i,j)
            for i, qa in enumerate(self.data)
            for j, a in enumerate(qa['answers'])
            if j <= 3
        ]
    def __len__(self):
        return len(self.qa_id_list)
    def make_example(self, idx):
        i,j = self.qa_id_list[idx]
        question = self.data['question'][i]

        context = self.data['ctxs'][i]

        answer = self.data['answers'][i][j]

        return (question, context,  answer)
    def __getitem__(self, idx):
        return self.make_example(idx)


def make_qa_s2s_batch(qa_list, tokenizer, max_len=64, max_a_len=360):
    q_ls = (q for q, c, a in qa_list)
    c_ls = (c for q, c, a in qa_list)
    a_ls = (a for q, c, a in qa_list)

    q_toks = tokenizer.batch_encode_plus(q_ls, c_ls, max_length=max_len, padding='max_length', truncation=True,
                                         return_tensors='pt')
    q_ids, q_mask = (
        torch.LongTensor(q_toks['input_ids']),
        torch.LongTensor(q_toks['attention_mask'])
    )
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=min(max_len, max_a_len), padding='max_length',
                                         truncation=True, return_tensors='pt')
    a_ids, a_mask = (
        torch.LongTensor(a_toks['input_ids']),
        torch.LongTensor(a_toks['attention_mask'])
    )
    labels = a_ids[:, 1:].contiguous().clone()
    labels[a_mask[:, 1:].contiguous() == 0] = -100

    #     print('q_ids shape',q_ids.shape)
    #     print('q_mask shape', q_mask.shape)
    #     print('a_ids shape', a_ids.shape)
    #     print('a_mask shape', a_mask.shape)
    #     print("labels shape", labels.shape)

    model_inputs = {
        'input_ids': q_ids,
        'attention_mask': q_mask,
        'decoder_input_ids': a_ids[:, :-1].contiguous(),
        'labels': labels,
    }
    return model_inputs

def data_loader(dataset, args, tokenizer):
    train_sampler = SequentialSampler(dataset)
    model_collate_fn = functools.partial(make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
    return data_loader

class ArgumentsS2S():
    def __init__(self, batch_size, max_length):
        self.batch_size = batch_size
        self.max_length = max_length

import lightning as L
class model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    def forward(self, batch_input):
        output = self.model(**batch_input)
        return output.loss, output.logits
    def training_step(self, batch, batch_idx):
        loss, output = self(batch)
        self.log('train_loss',loss, prog_bar=True, logger=True)
        return loss
    def validation_step(self, batch, batch_idx):
        loss, output = self(batch)
        self.log('val_loss',loss, prog_bar=True, logger=True)
        return loss
    def test_step(self, batch, batch_idx):
        loss, output = self(batch)
        self.log('test_loss',loss, prog_bar=True, logger=True)
        return loss
    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-4)
def fit(model, optimizer, tokenizer, train_loader, val_loader, epochs=3, device = 'cpu'):
    train_loss = 0
    val_loss = 0
    train_batch_count = 0
    val_batch_count = 0
    for epcoch in range(epochs):
        model.train()
        for batch in tqdm(train_loader, desc = 'Traning batches'):
            inputs_id = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            decoder_input_ids = batch['decoder_input_ids'].to(device)

            output = model(input_ids=inputs_id, attention_mask=attention_mask,
                           labels=labels, decoder_input_ids=decoder_input_ids)

            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()
            train_loss += output.loss.item()
            train_batch_count += 1

        #evaluation
        model.eval()
        for batch in tqdm(val_loader, desc='Traning batches'):
            inputs_id = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)

            output = model(inputs_id, attention_mask, labels, decoder_attention_mask)

            optimizer.zero_grad()
            output.loss.backward()
            optimizer.step()
            val_loss += output.loss.item()
            val_batch_count += 1

        print(f"{epcoch+1} --> Train loss: {train_loss/train_batch_count}, validation loss: {val_loss/val_batch_count}")

        model.save_pretrained("qa_model")
        tokenizer.save_pretrained('qa_tokenizer')
