from utils import *
train3 = datasets.load_from_disk("train_data_tokenized_all_splited_3_answers")
val3 = datasets.load_from_disk("valid_data_tokenized_all_splited_3_answers")

columns = ['input_ids', 'decoder_input_ids', 'attention_mask', 'decoder_attention_mask']
train3.set_format(type='torch', columns=columns)
val3.set_format(type='torch', columns=columns)

model = T5ForConditionalGeneration.from_pretrained("t5-base")
tokenizer = T5Tokenizer.from_pretrained("t5-base")

training_args = TrainingArguments(output_dir="model_3_ans/",
                                  per_device_train_batch_size=4,
                                  per_device_eval_batch_size=4,
                                  gradient_accumulation_steps=16,
                                  learning_rate=1e-4,
                                  num_train_epochs=2,
                                  logging_steps=500,
                                  run_name="QA_question_answering",
                                  evaluation_strategy="steps",
                                  save_steps=8367,
                                  optim='adamw_torch',
                                  report_to='none'
                                  )

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train3,
    eval_dataset= val3,
    data_collator=T2TDataCollator()
)

# Training
trainer.train()



