from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, T5TokenizerFast
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch
import functools
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
import math
from time import time
from datasets import load_dataset, Dataset
import evaluate
from rouge_score import rouge_scorer
import numpy as np
import pandas as pd
def replace_text(string):
    return string.replace("\'","").replace("\n","").replace("URL_0","").lower().strip()
def preprocess(ex, num_docs = 3):
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

        return (question, context, answer)
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
    labels = a_ids
    labels[labels == 0] = -100

    #     print('q_ids shape',q_ids.shape)
    #     print('q_mask shape', q_mask.shape)
    #     print('a_ids shape', a_ids.shape)
    #     print('a_mask shape', a_mask.shape)
    #     print("labels shape", labels.shape)

    model_inputs = {
        'input_ids': q_ids,
        'attention_mask': q_mask,
        'decoder_attention_mask': a_mask,
        'labels': labels,
    }
    return model_inputs

def data_loader(dataset, args, tokenizer):
    train_sampler = RandomSampler(dataset)
    model_collate_fn = functools.partial(make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length)
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
    return data_loader


class ArgumentsS2S():
    def __init__(self, batch_size, max_length = 256):
        self.batch_size = batch_size
        self.max_length = max_length

import lightning as L
class model(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base")
    def forward(self, batch_input):
        output = self.model(**batch_input)
        return output.loss, output.logits
    def training_step(self, batch, batch_idx):
        loss, output = self(batch)
        self.log('train_loss',loss, prog_bar=True, logger=True)
        print('train_loss:', loss)
        return loss
    def validation_step(self, batch, batch_idx):
        loss, output = self(batch)
        self.log('val_loss',loss, prog_bar=True, logger=True)
        print('val_lss:', loss)
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
            decoder_attention_mask = batch['decoder_attention_mask'].to(device)

            output = model(input_ids=inputs_id, attention_mask=attention_mask,
                           labels=labels, decoder_attention_mask=decoder_attention_mask)

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

def load_model(path):
    global model
    model = model()
    tokenizer = T5Tokenizer.from_pretrained('t5-base')
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    state_dict = checkpoint['state_dict']
    model.load_state_dict(state_dict)
    return model, tokenizer
def generate(model, tokenizer, question, context):
    inputs = tokenizer.batch_encode_plus(question, context, max_length=256, padding='max_length',
                                   truncation=True, return_tensors='pt').input_ids
    outputs = model.model.generate(inputs)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def generate_val(model, tokenizer, ex):
    inputs = tokenizer.batch_encode_plus(ex['question'], ex['context'], max_length=256, padding='max_length',
                                   truncation='only_second', return_tensors='pt').input_ids
    outputs = model.model.generate(inputs)
    ex['predict'] = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return ex

def eval(val1):
    dct = {'question': [], 'context': [], 'answer': []}
    qa_id_list = [
        (i, j)
        for i, qa in enumerate(val1)
        for j, a in enumerate(qa['answers'])
        if j <= 3
    ]
    for x in range(len(qa_id_list)):
        i,j = qa_id_list[x]
        dct['question'].append(val1['question'][i])

        dct['context'].append(val1['ctxs'][i])

        dct['answer'].append(val1['answers'][i][j])

    return Dataset.from_dict(dct)

def answers(val_1, tokenizer, model):
    answers = []
    model.freeze()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model1 = model.model.to(device)
    with torch.no_grad():
        for x in val_1:
            outputs = model1.generate(x['input_ids'].to(device))
            decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            answers += decoded
    return answers

def compute_Rouge_score(predict, reference):

    scorer = rouge_scorer.RougeScorer(['rougeL'])
    score_all = []
    for x, y in zip(predict, reference):
        score = list((scorer.score(str(x), str(y))).values())
        score=score[0]

        precision = score[0]
        recall = score[1]
        F1 = score[2]
        score_all.append([precision, recall, F1])

    df = pd.DataFrame(score_all, columns=['precision', 'recall', 'f1'])

    return np.mean(df)








