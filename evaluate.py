import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset
from random import randrange

peft_model_id = 'finetuned_model'
config = PeftConfig.from_pretrained(peft_model_id)

model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path, load_in_8bit=True, device_map={"":0})
tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

model = PeftModel.from_pretrained(model, peft_model_id, device_map={"":0})
model.eval()
print('Peft flan-t5-xl loaded')

dataset = load_dataset('opus_books', 'ca-nl')
dataset = dataset['train'].train_test_split(test_size=0.2)
sample = dataset['test'][randrange(len(dataset['test']))]
prompt = 'translate Catalan to Dutch (Standard): '

input_ids = tokenizer(prompt + sample['translation']['ca'], return_tensors='pt', truncation=True).input_ids.cuda()
outputs = model.generate(input_ids=input_ids, max_new_tokens=128, do_sample=True, top_p=0.9)
print(f"Source: {sample['translation']['ca']}\n{'---'*20}")
print(f"Gold translation: {sample['translation']['nl']}\n{'---'*20}")
print(f"Output by model: {tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)[0]}")