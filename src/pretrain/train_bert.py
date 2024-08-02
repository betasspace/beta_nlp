import json 
import os
from tqdm import tqdm

from transformers import BertTokenizerFast, BertConfig, BertForMaskedLM, DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers import pipeline
from datasets import concatenate_datasets, load_dataset, load_from_disk

model_path = "pretrained-bert"
vocab_size = 30_522
max_length = 512

dataset = load_from_disk("/home/tom/fssd/bert_dataset_longer_test")
d = dataset.train_test_split(test_size=0.1)
train_dataset = d["train"]
test_dataset = d["test"]
# train_dataset = load_from_disk("/home/tom/fsas/bert_dataset_longer_train")

tokenizer = BertTokenizerFast.from_pretrained(model_path)

# initialize the model with the config
model_config = BertConfig(vocab_size=vocab_size, max_position_embeddings=max_length)
model = BertForMaskedLM(config=model_config)
# initialize the data collator, randomly masking 20% (default is 15%) of the tokens # for the Masked Language Modeling (MLM) task
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer,
                                                mlm=True, mlm_probability=0.2)
training_args = TrainingArguments(
    output_dir=model_path,  # output directory to where save model checkpoint
    evaluation_strategy="epoch", # evaluate each `logging_steps` steps
    overwrite_output_dir=True, 
    num_train_epochs=10, # number of training epochs, feel free to tweak
    per_device_train_batch_size=20, # the training batch size, put it as high as your GPU memory fits
    gradient_accumulation_steps=4, # accumulating the gradients before updating the weights
    per_device_eval_batch_size=64, # evaluation batch size
    logging_steps=500, # evaluate, log and save model checkpoints every 1000 step
    save_steps=10000,
    save_total_limit=300,
    logging_dir="/home/tom/tensorboard_log",
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

trainer.train(resume_from_checkpoint=True)
# trainer.train(resume_from_checkpoint=f"{model_path}/checkpoint-50000")
# trainer.train()
