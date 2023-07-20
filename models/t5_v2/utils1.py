from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    EvalPrediction,
    DataCollator,
    Trainer,
    TrainingArguments)

from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5Model, T5ForConditionalGeneration, T5TokenizerFast
from typing import Dict, List, Optional
import dataclasses
from dataclasses import dataclass, field
import torch
max_input_length = 512
max_target_length = 64
def add_tokenizer(tokenizer):
    tokenizer.sep_token = '<sep>'
    tokenizer.add_tokens(['<sep>'])

checkpoint = "t5-base"
model = T5ForConditionalGeneration.from_pretrained(checkpoint)
tokenizer = T5TokenizerFast.from_pretrained(checkpoint)
add_tokenizer(tokenizer)
model.resize_token_embeddings(len(tokenizer))
def add_sep_token(example):
    example['answers'] = " <sep> ".join(example['answers'])
    return example
def add_eos_token(example):

    example['answers']=  example['answers'] + " </s>"
    return example

def convert_to_features(example_batch):

    input_encodings = tokenizer.batch_encode_plus(example_batch['question'],
                                                  example_batch['ctxs'],
                                                  max_length=max_input_length,
                                                  add_special_tokens=True,
                                                  truncation=True,
                                                  pad_to_max_length=True)

    target_encodings = tokenizer.batch_encode_plus(example_batch['answers'],
                                                   max_length=max_target_length,
                                                   add_special_tokens=True,
                                                   truncation=True, pad_to_max_length=True)

    encodings = {
        'input_ids': input_encodings['input_ids'],
        'attention_mask': input_encodings['attention_mask'],
        'decoder_input_ids': target_encodings['input_ids']
        ,'decoder_attention_mask': target_encodings['attention_mask']
    }

    return encodings





class T2TDataCollator():
  def __call__(self, batch: List) -> Dict[str, torch.Tensor]:
    """
    Take a list of samples from a Dataset and collate them into a batch.
    Returns:
    A dictionary of tensors
    """

    input_ids = torch.stack([example['input_ids'] for example in batch])
    lm_labels = torch.stack([example['decoder_input_ids'] for example in batch])
    lm_labels[lm_labels[:, :] == 0] = -100
    attention_mask = torch.stack([example['attention_mask'] for example in batch])
    decoder_attention_mask = torch.stack([example['decoder_attention_mask'] for example in batch])

    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'labels': lm_labels,
        'decoder_attention_mask': decoder_attention_mask
    }


class eli5dataset1(Dataset):
    def __init__(self, data):
        self.dataset = data
        self.qa_id_list = [
            (i,j)
            for i, qa in enumerate(self.dataset)
            for j, a in enumerate(qa['answers'])
            if j <= 3
        ]
    def __len__(self):
        return len(self.qa_id_list)
    def make_example(self, idx):
        i,j = self.qa_id_list[idx]
        question = self.dataset['question'][i]

        context = self.dataset['ctxs'][i]

        answer = self.dataset['answers'][i][j]

        return (question, context, answer)
    def __getitem__(self, idx):
        return self.make_example(idx)
