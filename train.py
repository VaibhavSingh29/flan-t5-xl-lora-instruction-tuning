from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset, load_from_disk
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, TaskType
from transformers import DataCollatorForSeq2Seq
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
import argparse
import torch

"""
    Indic language support on flan-t5-xl: Hindi, Bengali, Gujarati, Telugu, Tamil, Marathi, Malayalam, Oriya, Panjabi, Urdu, Kannada, Assamese
"""


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/flan-t5-xl",
        help="Model id to use for training.",
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of epochs to train for."
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=64,
        help="Batch size to use for training.",
    )
    parser.add_argument(
        "--lr", type=float, default=1e-3, help="Learning rate to use for training."
    )
    parser.add_argument(
        "--deepspeed", type=str, default=None, help="Path to deepspeed config file."
    )
    parser.add_argument(
        "--gradient_checkpointing",
        type=bool,
        default=True,
        help="Path to deepspeed config file.",
    )
    parser.add_argument(
        "--bf16",
        type=bool,
        default=True if torch.cuda.get_device_capability()[0] == 8 else False,
        help="Whether to use bf16.",
    )
    args = parser.parse_known_args()
    return args


def train_model(args):
    model_card = args.model_name
    source_language = "hi"
    target_language = "kn"
    prompt = "translate Hindi to Kannada: "

    dataset = load_from_disk(f"{source_language}_{target_language}_dataset")
    dataset = dataset["train"].train_test_split(test_size=0.2)

    print(f"Train size: {len(dataset['train'])}")
    print(f"Test size: {len(dataset['test'])}")

    tokenizer = AutoTokenizer.from_pretrained(model_card)

    def preprocess_function(examples):
        inputs = [prompt + example for example in examples[source_language]]
        targets = [example for example in examples[target_language]]
        model_inputs = tokenizer(
            inputs, text_target=targets, max_length=128, truncation=True
        )  # max_lengh to be changed
        return model_inputs

    tokenized_dataset = dataset.map(preprocess_function, batched=True)
    tokenized_dataset["train"].save_to_disk("tokenized_data/train")
    tokenized_dataset["test"].save_to_disk("tokenized_data/test")

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_card, load_in_8bit=True, device_map="auto"
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q", "v"],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )

    model = prepare_model_for_int8_training(model)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    label_pad_token_id = -100

    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8,
    )

    output_dir = "flan-t5-xl-lora"

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        # auto_find_batch_size=True,
        per_device_train_batch_size=64,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        deepspeed=args.deepspeed,
        gradient_checkpointing=args.gradient_checkpointing,
        logging_dir=f"{output_dir}/logs",
        logging_strategy="steps",
        logging_steps=500,
        save_strategy="no",
    )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_dataset["train"],
    )

    trainer.train()

    finetuned_model_id = "finetuned_model"
    trainer.model.save_pretrained(finetuned_model_id)
    tokenizer.save_pretrained(finetuned_model_id)


def main():
    args, _ = parse_arguments()
    train_model(args)


if __name__ == "__main__":
    main()
