from transformers import BartTokenizer, BartForConditionalGeneration
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
import torch
import functools
from tqdm import tqdm
from transformers import  get_linear_schedule_with_warmup
import math
from time import time
def replacetext(string):
    return string.replace("\\","").replace("\n","").replace("URL 0", "")
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
        i, j = self.qa_id_list[idx]
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


def make_qa_s2s_batch(qa_list, tokenizer, max_len=64, max_a_len=360, device='cpu'):
    q_ls = (q for q, a in qa_list)
    a_ls = (a for q, a in qa_list)

    q_toks = tokenizer.batch_encode_plus(q_ls, max_length=max_len, padding='only_second')
    q_ids, q_mask = (
        torch.LongTensor(q_toks['input_ids']).to(device),
        torch.LongTensor(q_toks['attention_mask']).to(device)
    )
    a_toks = tokenizer.batch_encode_plus(a_ls, max_length=min(max_len, max_a_len), padding='only_second')
    a_ids, a_mask = (
        torch.LongTensor(a_toks['input_ids']).to(device),
        torch.LongTensor(a_toks['attention_mask']).to(device)
    )
    labels = a_ids[:, 1:].contiguous().clone()
    #replace padding target token id's of the labels by -100, crossEntropy skip target label == -100
    labels[a_mask[:, 1:].contiguous() == 0] = -100

    model_inputs = {
        'input_ids': q_ids,
        'attention_mask': q_mask,
        'decoder_input_ids': a_ids[:, :-1].contiguous(),
        'labels': labels,
    }
    return model_inputs

def train_qa_s2s_epoch(model, dataset, tokenizer, optimizer, scheduler, args, e=0, curriculum=False):
    model.train()
    # make iterator
    if curriculum:
        train_sampler = SequentialSampler(dataset)
    else:
        train_sampler = RandomSampler(dataset)
    model_collate_fn = functools.partial(
        make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device=args.device
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=True)
    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    for step, batch_inputs in enumerate(epoch_iterator):
        pre_loss = model(**batch_inputs)[0]
        loss = pre_loss
        loss.backward()
        # optimizer
        if step % args.backward_freq == 0:
            optimizer.step()
            scheduler.step()
            model.zero_grad()
        # some printing within the epoch
        loc_loss += loss.item()
        loc_steps += 1
        if step % args.print_freq == 0 or step == 1:
            print(
                "{:2d} {:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                    e, step, len(dataset) // args.batch_size, loc_loss / loc_steps, time() - st_time,
                )
            )
            loc_loss = 0
            loc_steps = 0
def eval_qa_s2s_epoch(model, dataset, tokenizer, args):
    model.eval()
    # make iterator
    train_sampler = SequentialSampler(dataset)
    model_collate_fn = functools.partial(
        make_qa_s2s_batch, tokenizer=tokenizer, max_len=args.max_length, device=args.device
    )
    data_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler, collate_fn=model_collate_fn)
    epoch_iterator = tqdm(data_loader, desc="Iteration", disable=False)
    # accumulate loss since last print
    loc_steps = 0
    loc_loss = 0.0
    st_time = time()
    with torch.no_grad():
        for step, batch_inputs in enumerate(epoch_iterator):
            pre_loss = model(**batch_inputs)[0]
            loss = pre_loss
            loc_loss += loss.item()
            loc_steps += 1
            if step % args.print_freq == 0:
                print(
                    "{:5d} of {:5d} \t L: {:.3f} \t -- {:.3f}".format(
                        step, len(dataset) // args.batch_size, loc_loss / loc_steps, time() - st_time,
                    )
                )
    print("Total \t L: {:.3f} \t -- {:.3f}".format(loc_loss / loc_steps, time() - st_time,))


def train_qa_s2s(qa_s2s_model, qa_s2s_tokenizer, s2s_train_dset, s2s_valid_dset, s2s_args):
    s2s_optimizer = torch.optim.AdamW(qa_s2s_model.parameters(), lr=s2s_args.learning_rate, eps=1e-8)
    s2s_scheduler = get_linear_schedule_with_warmup(
        s2s_optimizer,
        num_warmup_steps=400,
        num_training_steps=(s2s_args.num_epochs + 1) * math.ceil(len(s2s_train_dset) / s2s_args.batch_size),
    )
    for e in range(s2s_args.num_epochs):
        train_qa_s2s_epoch(
            qa_s2s_model,
            s2s_train_dset,
            qa_s2s_tokenizer,
            s2s_optimizer,
            s2s_scheduler,
            s2s_args,
            e,
            curriculum=(e == 0),
        )
        m_save_dict = {
            "model": qa_s2s_model.state_dict(),
            "optimizer": s2s_optimizer.state_dict(),
            "scheduler": s2s_scheduler.state_dict(),
        }
        print("Saving model {}".format(s2s_args.model_save_name))
        eval_qa_s2s_epoch(qa_s2s_model, s2s_valid_dset, qa_s2s_tokenizer, s2s_args)
        torch.save(m_save_dict, "models/bart/model_saved/{}_{}.pth".format(s2s_args.model_save_name, e))

def qa_s2s_generate(
    question_doc,
    qa_s2s_model,
    qa_s2s_tokenizer,
    num_answers=1,
    num_beams=None,
    min_len=64,
    max_len=256,
    do_sample=False,
    temp=1.0,
    top_p=None,
    top_k=None,
    max_input_length=512,
    device="cpu",
):
    model_inputs = make_qa_s2s_batch([(question_doc, "A")], qa_s2s_tokenizer, max_input_length, device=device)
    n_beams = num_answers if num_beams is None else max(num_beams, num_answers)
    generated_ids = qa_s2s_model.generate(
        input_ids=model_inputs["input_ids"],
        attention_mask=model_inputs["attention_mask"],
        min_length=min_len,
        max_length=max_len,
        do_sample=do_sample,
        early_stopping=True,
        num_beams=1 if do_sample else n_beams,
        temperature=temp,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=qa_s2s_tokenizer.eos_token_id,
        no_repeat_ngram_size=3,
        num_return_sequences=num_answers,
        decoder_start_token_id=qa_s2s_tokenizer.bos_token_id,
    )
    return [qa_s2s_tokenizer.decode(ans_ids, skip_special_tokens=True).strip() for ans_ids in generated_ids]