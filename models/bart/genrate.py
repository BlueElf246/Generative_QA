from utils import *
import torch
# load model
model, optimizer, scheduler = torch.load("model_saved/*.pth")
#
question = 'abc'
context = 'xyz'
inputs = f"question: {question} context: {context}"

outputs = qa_s2s_generate(inputs, model, optimizer)