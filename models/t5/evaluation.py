from utils import *

model,tokenizer = load_model(path="/Users/datle/Desktop/code_structure/models/t5/lightning_logs/version_0/checkpoints/epoch=0-step=4119.ckpt")

val = load_dataset('json', data_files="/Users/datle/Downloads/ELI5_val.jsonl")['train']

val1 = val.map(lambda ex: preprocess(ex,num_docs=3), remove_columns=['question_id'])

ds_val = eval(val1)

predicted =ds_val.add_column('predicted', answers)

result_rouge = compute_Rouge_score(predicted['predicted'], predicted['answer'])
print(result_rouge)


