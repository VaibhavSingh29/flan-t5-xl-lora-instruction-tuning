from datasets import DatasetDict, Dataset


def main():
    with open("./hi-kn/train.hi", "r", encoding="utf-8") as source_language_file:
        source_sentences = [line.strip() for line in source_language_file.readlines()]

    with open("./hi-kn/train.kn", "r", encoding="utf-8") as target_language_file:
        target_sentences = [line.strip() for line in target_language_file.readlines()]

    source_language = "hi"
    target_language = "kn"

    data = {
        "train": {source_language: source_sentences, target_language: target_sentences},
        "validation": {source_language: [], target_language: []},
        "test": {source_language: [], target_language: []},
    }

    custom_dataset = DatasetDict()
    for split in data:
        custom_dataset[split] = Dataset.from_dict(data[split])

    custom_dataset.save_to_disk(f"{source_language}_{target_language}_dataset")


if __name__ == "__main__":
    main()
