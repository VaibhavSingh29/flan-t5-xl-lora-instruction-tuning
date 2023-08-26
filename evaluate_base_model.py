import torch
from peft import PeftModel, PeftConfig
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
from random import randrange

model_card = "google/flan-t5-xl"

model = AutoModelForSeq2SeqLM.from_pretrained(
    model_card, load_in_8bit=True, device_map={"": 0}
)
tokenizer = AutoTokenizer.from_pretrained(model_card)

model.eval()
print("Base flan-t5-xl loaded")

source_language = "hi"
target_language = "kn"
prompt = "translate Hindi to Kannada: "
dataset = load_from_disk(f"{source_language}_{target_language}_dataset")
dataset = dataset["train"].train_test_split(test_size=0.2)
sample = dataset["test"][0]


input_ids = tokenizer(
    prompt + sample[source_language], return_tensors="pt", truncation=True
).input_ids.cuda()
outputs = model.generate(
    input_ids=input_ids, max_new_tokens=128, do_sample=True, top_p=0.9
)
print(f"Source: {sample[source_language]}\n{'---'*20}")
print(f"Gold translation: {sample[target_language]}\n{'---'*20}")
print(
    f"Output by model: {tokenizer.batch_decode(outputs.detach().cpu().numpy(), skip_special_tokens=True)}"
)
