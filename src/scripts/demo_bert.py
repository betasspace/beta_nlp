
import json 
import os

from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import pipeline
from tokenizers import BertWordPieceTokenizer
from datasets import concatenate_datasets, load_dataset, load_from_disk

bookcorpus = load_dataset("bookcorpus", split="train")
# bookcorpus.save_to_disk('/Users/bytedance/workspace/test/multi_gpu/dataset/bookcorpus')

wiki = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
# wiki.save_to_disk('/Users/bytedance/workspace/test/multi_gpu/dataset/wikipedia')

wiki1 = load_from_disk("/root/autodl-tmp/cache/bert_data/wikipedia/20220301.en")

# 仅保留 'text' 列
wiki = wiki.remove_columns([col for col in wiki.column_names if col != "text"])
dataset = concatenate_datasets([bookcorpus, wiki])
# 将数据集合切分为 90% 用于训练，10% 用于测试 d = dataset.train_test_split(test_size=0.1)

def dataset_to_text(dataset, output_filename="data.txt"):
    """Utility function to save dataset text to disk,
     useful for using the texts to train the tokenizer
     (as the tokenizer accepts files)"""
    with open(output_filename, "w") as f:
       for t in dataset["text"]:print(t, file=f)
# save the training set to train.txt
dataset_to_text(dataset["train"], "train.txt") 
# save the testing set to test.txt 
dataset_to_text(dataset["test"], "test.txt")

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
# initialize the WordPiece tokenizer
tokenizer = BertWordPieceTokenizer() # train the tokenizer

tokenizer.train(files=files, vocab_size=vocab_size, special_tokens=special_tokens) 
# enable truncation up to the maximum 512 tokens 
tokenizer.enable_truncation(max_length=max_length)

model_path = "pretrained-bert"

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
# when the tokenizer is trained and configured, load it as BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained(model_path)


def encode_with_truncation(examples):
    """Mapping function to tokenize the sentences passed with truncation"""
    return tokenizer(examples["text"], truncation=True, padding="max_length",
                     max_length=max_length, return_special_tokens_mask=True)
def encode_without_truncation(examples):
    """Mapping function to tokenize the sentences passed without truncation"""
    return tokenizer(examples["text"], return_special_tokens_mask=True)

# the encode function will depend on the truncate_longer_samples variable
encode = encode_with_truncation if truncate_longer_samples else encode_without_truncation # tokenizing the train dataset
train_dataset = dataset["train"].map(encode, batched=True)
# tokenizing the testing dataset
test_dataset = dataset["test"].map(encode, batched=True) 
if truncate_longer_samples:
  # remove other columns and set input_ids and attention_mask as PyTorch tensors
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    test_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
else:
  # remove other columns, and remain them as Python lists
  test_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])
  train_dataset.set_format(columns=["input_ids", "attention_mask", "special_tokens_mask"])



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
    train_dataset = train_dataset.map(group_texts, batched=True,
                                      desc=f"Grouping texts in chunks of {max_length}")
    test_dataset = test_dataset.map(group_texts, batched=True,
                                    desc=f"Grouping texts in chunks of {max_length}")
    # convert them from lists to torch tensors
    train_dataset.set_format("torch")
    test_dataset.set_format("torch")


# initialize the model with the config
model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)
# initialize the data collator, randomly masking 20% (default is 15%) of the tokens # for the Masked Language Modeling (MLM) task
data_collator = DataCollatorForLanguageModeling(
tokenizer=tokenizer, mlm=True, mlm_probability=0.2 )
training_args = TrainingArguments(
    output_dir=model_path,  # output directory to where save model checkpoint
    eval_strategy="epoch", # evaluate each `logging_steps` steps
    overwrite_output_dir=True, 
    num_train_epochs=10, # number of training epochs, feel free to tweak
    per_device_train_batch_size=10, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=8, # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64, # evaluation batch size
    logging_steps=1000, # evaluate, log and save model checkpoints every 1000 step
    save_steps=100000,
    # load_best_model_at_end=True,  # whether to load the best model (in terms of loss)
)


trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
)
# train the model
# whether you don't have much space so you
# let only  3 model weights saved in the disk

trainer.train()



# load the model checkpoint
model = BertForMaskedLM.from_pretrained(os.path.join(model_path, "checkpoint-10000")) # load the tokenizer
tokenizer = BertTokenizerFast.from_pretrained(model_path)
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)
# perform predictions
examples = [
  "Today's most trending hashtags on [MASK] is Donald Trump",
  "The [MASK] was cloudy yesterday, but today it's rainy.",
]
for example in examples:
    for prediction in fill_mask(example):
        print(f"{prediction['sequence']}, confidence: {prediction['score']}")
    print("="*50)