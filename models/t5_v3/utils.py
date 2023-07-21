#import model
from transformers import T5ForConditionalGeneration, T5TokenizerFast, T5Tokenizer, BartTokenizer
#import loading dataset
import torch
from torch.utils.data import Dataset, RandomSampler, SequentialSampler
from datasets import load_dataset
from datasets import Dataset
#import Trainer
from transformers import Trainer, TrainingArguments, DataCollator
# bla bla
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
from rouge_score import rouge_scorer
import datasets
from transformers import TrainerCallback
max_input_length = 512
max_target_length = 64
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
def replace_text(string):
    return string.replace("\'","").replace("\n","").replace("URL_0","").lower().strip()
def preprocess(ex, num_docs = 3):
    ex['question'] = replace_text(ex['question'])
    context = ex['ctxs'][:num_docs]
    if type((context[0])) == list:
        context = [k[0] for k in context]
    context = replace_text(' '.join(context))
    ex['ctxs'] = context
    ex['answers'] = [replace_text(i) for i in ex['answers']]
    return ex

def tokenize_input(example, tokenizer):
    return tokenizer(example['question'], example['ctxs'],
                                max_length=max_input_length,
                                add_special_tokens=True,
                                truncation='only_second',
                                return_attention_mask=True,
                                padding='max_length')
def tokenize_output(example, tokenizer):
    result = tokenizer(example['answers'],
                                max_length=max_target_length,
                                add_special_tokens=True,
                                truncation=True,
                                return_attention_mask=True,
                                padding='max_length')
    example['decoder_input_ids'] = result['input_ids']
    example['decoder_attention_mask'] = result['attention_mask']
    return example
def decompose(example, dct):
    input_ids = example['input_ids']
    attention_mask = example['attention_mask']
    for x,y in zip(example['decoder_input_ids'], example['decoder_attention_mask']):
        dct['input_ids'].append(input_ids)
        dct['attention_mask'].append(attention_mask)
        dct['decoder_input_ids'].append(x)
        dct['decoder_attention_mask'].append(y)

def filter_answer(ex, num_answers=3):
    ex['decoder_input_ids'] = ex['decoder_input_ids'][:num_answers]
    ex['decoder_attention_mask'] = ex['decoder_attention_mask'][:num_answers]
    return ex

def decompose(example, dct):
    input_ids = example['input_ids']
    attention_mask = example['attention_mask']
    for x,y in zip(example['decoder_input_ids'], example['decoder_attention_mask']):
        dct['input_ids'].append(input_ids)
        dct['attention_mask'].append(attention_mask)
        dct['decoder_input_ids'].append(x)
        dct['decoder_attention_mask'].append(y)

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
        'decoder_attention_mask': decoder_attention_mask}

class CustomLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            print("Custom log information:")
            for key, value in logs.items():
                print(f"{key}: {value}")
            print("=" * 30)

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

def tokenize(example, tokenizer):
    input_encodings = tokenizer(example['question'], example['ctxs'],
                                max_length=max_input_length,
                                add_special_tokens=True,
                                truncation='only_second',
                                return_attention_mask=True,
                                padding='max_length')
    encodings = {
        'input_ids': input_encodings['input_ids'],
        'answer': ". ".join(example['answers'])
    }

def generate(example, model, tokenizer):
    inputs = torch.LongTensor(example['input_ids']).to(device)
    outputs = model.generate(inputs, do_sample=True, max_length=64)
    answers = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    example['predicted'] = answers
    return example


def generate_in_use(ques, ctxs, model, tokenizer):
    inputs = tokenizer(ques, ctxs,
                       max_length=max_input_length,
                       add_special_tokens=True,
                       truncation='only_second',
                       return_attention_mask=True,
                       padding='max_length', return_tensors='pt').input_ids

    outputs = model.generate(inputs, max_length=128, do_sample=True)
    answers = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(ques)
    print(answers)