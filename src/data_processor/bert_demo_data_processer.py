import json 
import os
from tqdm import tqdm

from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import pipeline
from tokenizers import BertWordPieceTokenizer
from datasets import concatenate_datasets, load_dataset, load_from_disk, save_to_disk

# bookcorpus = load_from_disk("/home/tom/fsas/bookcorpus")
# wiki = load_dataset("parquet", data_dir="/home/tom/fsas/wikipedia/data/20220301.en/", split="train")
# wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
# dataset = concatenate_datasets([bookcorpus, wiki])
# d = dataset.train_test_split(test_size=0.1)

special_tokens = [
  "[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<S>", "<T>"
]
# if you want to train the tokenizer on both sets # files = ["train.txt", "test.txt"]
# training the tokenizer on the training set files = ["train.txt"]
files = ["train.txt", "test.txt"]
# 30,522 vocab is BERT's default vocab size, feel free to tweak
vocab_size = 30_522
# maximum sequence length, lowering will result to faster training (when increasing batch size) 
max_length = 512

# whether to truncate
truncate_longer_samples = False
# whether to train tokenizer
need_2_train_tokenizer = False
need_2_tokenize = False

model_path = "pretrained-bert"

if need_2_train_tokenizer:
    def dataset_to_text(dataset, output_filename="data.txt"):
        """Utility function to save dataset text to disk,
         useful for using the texts to train the tokenizer
         (as the tokenizer accepts files)"""
        with open(output_filename, "w") as fout:
            for t in tqdm(dataset):
                fout.write(t["text"]+"\n")
                # print(t, file=f)
    # # save the training set to train.txt
    # dataset_to_text(d["train"], "train.txt") 
    # # save the testing set to test.txt 
    # dataset_to_text(d["test"], "test.txt")

    # initialize the WordPiece tokenizer
    tokenizer = BertWordPieceTokenizer()
    # train the tokenizer
    tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens) 
    # enable truncation up to the maximum 512 tokens 
    tokenizer.enable_truncation(max_length=max_length)
    # make the directory if not already there
    if not os.path.isdir(model_path):
       os.mkdir(model_path)
    # save the tokenizer
    tokenizer.save_model(model_path)
    # dumping some of the tokenizer config to config file,
    # including special tokens, whether to lower case and the maximum sequence length 
    with open(os.path.join(model_path, "config.json"), "w") as f:
        tokenizer_cfg = {
            "do_lower_case": True, 
            "unk_token": "[UNK]",
            "sep_token": "[SEP]",
            "pad_token": "[PAD]", 
            "cls_token": "[CLS]", 
            "mask_token": "[MASK]", 
            "model_max_length": max_length, 
            "max_len": max_length,
        }
        json.dump(tokenizer_cfg, f)
tokenizer = BertTokenizerFast.from_pretrained(model_path)


def encode_with_truncation(examples):
    """Mapping function to tokenize the sentences passed with truncation"""
    return tokenizer(examples["text"], truncation=True, padding="max_length",
                     max_length=max_length, return_special_tokens_mask=True)
def encode_without_truncation(examples):
    """Mapping function to tokenize the sentences passed without truncation"""
    return tokenizer(examples["text"], return_special_tokens_mask=True)

if need_2_tokenize:
    train_dataset = load_dataset("/home/tom/fsas/bert_dataset", split="train")
    test_dataset = load_dataset("/home/tom/fsas/bert_dataset", split="test")
    # the encode function will depend on the truncate_longer_samples variable
    encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation # tokenizing the train dataset
    train_dataset = train_dataset.map(encode, batched=True, desc="tokenize text in train set")
    # tokenizing the testing dataset
    test_dataset = test_dataset.map(encode, batched=True, desc="tokenize text in test set")
    if truncate_longer_samples:
        # remove other columns and set input_ids and attention_mask as PyTorch tensors
        train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
        test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    else:
        # remove other columns, and remain them as Python lists
        test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
        train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
    train_dataset.save_to_disk("/home/tom/fsas/bert_dataset_train/")
    test_dataset.save_to_disk("/home/tom/fsas/bert_dataset_test")
else:
    train_dataset = load_from_disk("/home/tom/fsas/bert_dataset_tokenized/train")
    test_dataset = load_from_disk("/home/tom/fsas/bert_dataset_tokenized/test")

from itertools import chain
# Main data processing function that will concatenate all texts from our dataset 
# and generate chunks of max_seq_length.

def group_texts(examples):
    # Concatenate all texts.
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()} 
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the small remainder, we could add padding if the model supported it instead of # this drop, you can customize this part to your needs.
    if total_length >= max_length:
        total_length = (total_length // max_length) * max_length 
    # Split by chunks of max_len.
    result = {
        k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
        for k, t in concatenated_examples.items()
        }
    return result

# Note that with `batched=True`, this map processes 1,000 texts together, so group_texts throws
# away a remainder for each of those groups of 1,000 texts. You can adjust that batch_size here but # a higher value might be slower to preprocess.
#
# To speed up this part, we use multiprocessing. See the documentation of the map method
#for more information:
# https://huggingface.co/docs/datasets/package_reference/main_classes.html#datasets.Dataset.map
if not truncate_longer_samples:
    train_dataset = train_dataset.map(group_texts, batched=True, num_proc=14,
                                      desc=f"Grouping texts in chunks of {max_length}",
                                      load_from_cache_file=True)
    test_dataset = test_dataset.map(group_texts, batched=True, num_proc=14,
                                    desc=f"Grouping texts in chunks of {max_length}",
                                    load_from_cache_file=True)
    # convert them from lists to torch tensors
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")
    test_dataset.save_to_disk("/home/tom/fsas/bert_dataset_longer_test_2")
    train_dataset.save_to_disk("/home/tom/fsas/bert_dataset_longer_train")
