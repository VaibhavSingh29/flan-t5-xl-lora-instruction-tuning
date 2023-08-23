from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments

"""
    Indic language support on flan-t5-xl: Hindi, Bengali, Gujarati, Telugu, Tamil, Marathi, Malayalam, Oriya, Panjabi, Urdu, Kannada, Assamese
"""
model_card = 'google/flan-t5-small'                                    
dataset = load_dataset('opus_books', 'ca-nl')                      # dummy dataset, to be replaced with the actual dataloader when training on DGX
print(len(dataset))
dataset = dataset['train'].train_test_split(test_size=0.2)
print(dataset['train'][0])
print(f"Train size: {len(dataset['train'])}")
print(f"Train size: {len(dataset['test'])}")

tokenizer = AutoTokenizer.from_pretrained(model_card)

source_language = 'ca'
target_language = 'nl'
prompt = 'translate Catalan to Dutch (Standard): '

def preprocess_function(examples):
    inputs = [prompt + example[source_language] for example in examples['translation']]
    targets = [example[target_language] for example in examples['translation']]
    model_inputs = tokenizer(inputs, text_target=targets, max_length=128, truncation=True)      # max_lengh to be changed
    return model_inputs

tokenized_dataset = dataset.map(preprocess_function, batched=True)
tokenized_dataset['train'].save_to_disk('data/train')
tokenized_dataset['test'].save_to_disk('data/test')

model = AutoModelForSeq2SeqLM.from_pretrained(model_card, load_in_8bit=True, device_map='auto')

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=['q', 'v'],
    lora_dropout=0.05,
    bias='none',
    task_type=TaskType.SEQ_2_SEQ_LM
)

model = prepare_model_for_int8_training(model)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

label_pad_token_id = -100

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    model=model,
    label_pad_token_id=label_pad_token_id,
    pad_to_multiple_of=8
)

output_dir = 'flan-t5-xl-lora'

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    auto_find_batch_size=True,
    learning_rate=1e-3,
    num_train_epochs=5,
    logging_dir=f"{output_dir}/logs",
    logging_strategy='steps',
    logging_steps=500,
    save_strategy='no'
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=tokenized_dataset['train']
)

trainer.train()

finetuned_model_id = 'finetuned_model'
trainer.model.save_pretrained(finetuned_model_id)
tokenizer.save_pretrained(finetuned_model_id)