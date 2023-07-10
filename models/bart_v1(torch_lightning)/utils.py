from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch
import functools
import lightning as L
import math
from time import time
from datasets import load_dataset
def replacetext(string):
    return string.replace("\\","").replace("\n","").replace("URL_0","")
class eli5dataset(Dataset):
    def __init__(self, data, num_docs):
        self.data = data
        self.qa_id_list = [
            (i,j)
            for i, qa in enumerate(self.data)
            for j, a in enumerate(qa['answers'])
        ]
        self.num_docs = num_docs
    def __len__(self):
        return len(self.qa_id_list)
    def make_example(self, idx):
        i,j = self.qa_id_list[idx]
        question = self.data['question'][i]
        question = replacetext(question)

        context = self.data['ctxs'][i][:self.num_docs]
        context = [k[0] for k in context]
        context = ' '.join(context)
        context = replacetext(context)

        answer = self.data['answers'][i][j]

        inputs = 'question: {} context: {}'.format(question.lower(), context.lower().strip())

        outputs = answer

        return (inputs, outputs)
    def __getitem__(self, idx):
        return self.make_example(idx)

def make_qa_s2s_batch(qa_list, tokenizer, max_len=64, max_a_len=360):
    q_ls = (q for q, a in qa_list)
    a_ls = (a for q, a in qa_list)

    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=max_len, padding='max_length', truncation=True)
    q_ids, q_mask = (
        torch.LongTensor(q_toks['input_ids']),
        torch.LongTensor(q_toks['attention_mask'])
    )
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=min(max_len, max_a_len), padding='max_length', truncation=True)
    a_ids, a_mask = (
        torch.LongTensor(a_toks['input_ids']),
        torch.LongTensor(a_toks['attention_mask'])
    )
    labels = a_ids[:, 1:].contiguous().clone()
    labels[a_mask[:, 1:].contiguous() == 0] = -100

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
class bart_model(L.LightningModule):
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