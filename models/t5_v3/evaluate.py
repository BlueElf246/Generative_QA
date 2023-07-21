from utils  import  *
path_val = "C:/Users/Admin/Downloads/ELI5_val.jsonl"
dataset_val = load_dataset('json', data_files = path_val)
val = dataset_val['train']

model_path = "C:/Users/Admin/OneDrive/Desktop/jupyter/model_3_ans/checkpoint-8367"
model = T5ForConditionalGeneration.from_pretrained(model_path)
tokenizer = T5Tokenizer.from_pretrained("t5-base")

val1 = val.map(lambda ex: preprocess(ex,num_docs=4), remove_columns=['question_id'])
val3 = val1.map(lambda ex: tokenize(ex, tokenizer), remove_columns=['question','ctxs','answers'])

model.to(device)

result1 = val3.map(lambda ex: generate(ex, model, tokenizer), batch_size=24, batched=True)

score = compute_Rouge_score(result1['predicted'], result1['answer'])
print(score)
