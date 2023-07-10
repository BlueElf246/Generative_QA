from utils import *
import torch
from transformers import BartTokenizer, BartForConditionalGeneration
# load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
params_dict = torch.load("model_saved/*.pth")
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
model = BartForConditionalGeneration.from_pretrained("facebook/bart-base").to(device)
model.load_state_dict(params_dict['model'])
#
question = 'abc'
context = 'xyz'
inputs = f"question: {question} context: {context}"

outputs = qa_s2s_generate(inputs, model, tokenizer)