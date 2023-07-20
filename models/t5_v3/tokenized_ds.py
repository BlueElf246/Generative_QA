from utils import *
tokenizer = T5Tokenizer.from_pretrained("t5-base")

path_train = "C:/Users/Admin/Downloads/ELI5.jsonl"
path_val = "C:/Users/Admin/Downloads/ELI5_val.jsonl"

dataset_train = load_dataset('json', data_files = path_train)
dataset_val = load_dataset('json', data_files = path_val)

train = dataset_train['train']
val = dataset_val['train']

train1 = train.map(lambda ex: preprocess(ex,num_docs=4), remove_columns = ['question_id'])
val1 = val.map(lambda ex: preprocess(ex,num_docs=4), remove_columns = ['question_id'])

train2 = train1.map(lambda ex: tokenize_input(ex, tokenizer), batched=True, remove_columns = ['question','ctxs'])
val2 = val1.map(lambda ex: tokenize_input(ex, tokenizer), batched=True, remove_columns = ['question','ctxs'])

train3 = train2.map(lambda ex: tokenize_output(ex, tokenizer), remove_columns = ['answers'])
val3 = val2.map(lambda ex: tokenize_output(ex, tokenizer), remove_columns = ['answers'])

# columns = ['input_ids', 'decoder_input_ids', 'attention_mask', 'decoder_attention_mask']
# train3.set_format(type='torch', columns=columns)
# val3.set_format(type='torch', columns=columns)

train3 = train3.map(lambda ex: filter_answer(ex, num_answers=3))
val3 = val3.map(lambda ex: filter_answer(ex, num_answers=3))

dct_train={'input_ids':[], 'attention_mask':[], "decoder_input_ids":[], "decoder_attention_mask":[]}
train3.map(lambda ex: decompose(ex, dct_train))
dct_val={'input_ids':[], 'attention_mask':[], "decoder_input_ids":[], "decoder_attention_mask":[]}
val3.map(lambda ex: decompose(ex, dct_val))

train3 = Dataset.from_dict(dct_train)
val3 = Dataset.from_dict(dct_val)

columns = ['input_ids', 'decoder_input_ids', 'attention_mask', 'decoder_attention_mask']
train3.set_format(type='torch', columns=columns)
val3.set_format(type='torch', columns=columns)

train3.save_to_disk("train_data_tokenized_all_splited_3_answers")
val3.save_to_disk("valid_data_tokenized_all_splited_3_answers")



