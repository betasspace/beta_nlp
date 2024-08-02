#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# File              : gcn_imdb_sentence_classification.py
# Author            : admin <admin>
# Date              : 31.12.2021
# Last Modified Date: 06.01.2022
# Last Modified By  : admin <admin>

import argparse

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

import torchtext
from torchtext.datasets import IMDB
# pip install torchtext 安装指令
from torchtext.datasets.imdb import NUM_LINES
from torchtext.data import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import to_map_style_dataset

import sys
import os
import logging
logging.basicConfig(
    level=logging.WARN,
    stream=sys.stdout,
    format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
)

parser = argparse.ArgumentParser(description='GCN_DDP')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=3, type=int,
                    metavar='N')
parser.add_argument('-wd', '--weight_decay', default=1e-3, type=float,
                    metavar='N')
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
args = parser.parse_args()

VOCAB_SIZE = 15000
# 第一期： 编写GCNN模型代码
class GCNN(nn.Module):
    def __init__(self, vocab_size=VOCAB_SIZE, embedding_dim=64, num_class=2):
        super(GCNN, self).__init__()

        self.embedding_table = nn.Embedding(vocab_size, embedding_dim)
        nn.init.xavier_uniform_(self.embedding_table.weight)

        self.conv_A_1 = nn.Conv1d(embedding_dim, 64, 15, stride=7)
        self.conv_B_1 = nn.Conv1d(embedding_dim, 64, 15, stride=7)

        self.conv_A_2 = nn.Conv1d(64, 64, 15, stride=7)
        self.conv_B_2 = nn.Conv1d(64, 64, 15, stride=7)

        self.output_linear1 = nn.Linear(64, 128)
        self.output_linear2 = nn.Linear(128, num_class)

    def forward(self, word_index):
        # 定义GCN网络的算子操作流程，基于句子单词ID输入得到分类logits输出

        # 1. 通过word_index得到word_embedding
        # word_index shape:[bs, max_seq_len]
        word_embedding = self.embedding_table(word_index) #[bs, max_seq_len, embedding_dim]

        # 2. 编写第一层1D门卷积模块
        word_embedding = word_embedding.transpose(1, 2) #[bs, embedding_dim, max_seq_len]
        A = self.conv_A_1(word_embedding)
        B = self.conv_B_1(word_embedding)
        H = A * torch.sigmoid(B) #[bs, 64, max_seq_len]

        A = self.conv_A_2(H)
        B = self.conv_B_2(H)
        H = A * torch.sigmoid(B) #[bs, 64, max_seq_len]

        # 3. 池化并经过全连接层
        pool_output = torch.mean(H, dim=-1) #平均池化，得到[bs, 64]
        linear1_output = self.output_linear1(pool_output)
        logits = self.output_linear2(linear1_output) #[bs, 2]

        return logits


class TextClassificationModel(nn.Module):
    """ 简单版embeddingbag+DNN模型 """

    def __init__(self, vocab_size=VOCAB_SIZE, embed_dim=64, num_class=2):
        super(TextClassificationModel, self).__init__()
        self.embedding = nn.EmbeddingBag(vocab_size, embed_dim, sparse=False)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, token_index):
        embedded = self.embedding(token_index) # shape: [bs, embedding_dim]
        return self.fc(embedded)



# step2 构建IMDB DataLoader

BATCH_SIZE = 64

def yield_tokens(train_data_iter, tokenizer):
    for i, sample in enumerate(train_data_iter):
        label, comment = sample
        yield tokenizer(comment)

train_data_iter = IMDB(root='.data', split='train') # Dataset类型的对象
tokenizer = get_tokenizer("basic_english")
vocab = build_vocab_from_iterator(yield_tokens(train_data_iter, tokenizer), min_freq=20, specials=["<unk>"])
vocab.set_default_index(0)
print(f"单词表大小: {len(vocab)}")

def collate_fn(batch):
    """ 对DataLoader所生成的mini-batch进行后处理 """
    target = []
    token_index = []
    max_length = 0
    for i, (label, comment) in enumerate(batch):
        if label == 1:
            target.append(0)
        elif label == 2:
            target.append(1)
        else:
            raise Exception(f"label: {label} is unexpect!")

        tokens = tokenizer(comment)
        token_index.append(vocab(tokens))
        if len(tokens) > max_length:
            max_length = len(tokens)

    token_index = [index + [0]*(max_length-len(index)) for index in token_index]
    return (torch.tensor(target).to(torch.int64), torch.tensor(token_index).to(torch.int32))


# step3 编写训练代码
def train(train_dataset, eval_dataset, model, optimizer, num_epoch, log_step_interval, save_step_interval, eval_step_interval, save_path, resume=""):
    """ 此处data_loader是map-style dataset """
    start_epoch = 0
    start_step = 0
    if resume != "":
        #  加载之前训过的模型的参数文件
        logging.warning(f"loading from {resume}")
        checkpoint = torch.load(resume, map_location="cuda:0")
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        start_step = checkpoint['step']
    
    model = nn.parallel.DistributedDataParallel(model.cuda(args.local_rank), device_ids=[args.local_rank]) # 模型拷贝，放入DP中

    train_sampler = DistributedSampler(train_dataset)
    train_data_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn, sampler=train_sampler)

    eval_data_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, collate_fn=collate_fn)


    for epoch_index in range(start_epoch, num_epoch):
        ema_loss = 0.
        num_batches = len(train_data_loader)

        train_sampler.set_epoch(epoch_index) # 每个epoch修改随机种子

        for batch_index, (target, token_index) in enumerate(train_data_loader):
            optimizer.zero_grad()
            step = num_batches*(epoch_index) + batch_index + 1

            # 数据拷贝
            # tensor.cuda() 需要重新赋值，nn.module.cuda()不需要赋值
            token_index = token_index.cuda(args.local_rank) 
            target = target.cuda(args.local_rank)

            logits = model(token_index)
            bce_loss = F.binary_cross_entropy(torch.sigmoid(logits), F.one_hot(target, num_classes=2).to(torch.float32))
            ema_loss = 0.9*ema_loss + 0.1*bce_loss
            bce_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 0.1)
            optimizer.step()

            if step % log_step_interval == 0:
                logging.warning(f"epoch_index: {epoch_index}, batch_index: {batch_index}, ema_loss: {ema_loss.item()}, bce_loss: {bce_loss.item()}")

            if step % save_step_interval == 0 and args.local_rank == 0:
                os.makedirs(save_path, exist_ok=True)
                save_file = os.path.join(save_path, f"step_{step}.pt")
                torch.save({
                    'epoch': epoch_index,
                    'step': step,
                    'model_state_dict': model.module.state_dict(), # DP.module == model
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': bce_loss,
                }, save_file)
                logging.warning(f"checkpoint has been saved in {save_file}")

            if step % eval_step_interval == 0:
                logging.warning("start to do evaluation...")
                model.eval()
                ema_eval_loss = 0
                total_acc_account = 0
                total_account = 0
                for eval_batch_index, (eval_target, eval_token_index) in enumerate(eval_data_loader):
                    total_account += eval_target.shape[0]
                    eval_logits = model(eval_token_index)
                    total_acc_account += (torch.argmax(eval_logits, dim=-1) == eval_target).sum().item()
                    eval_bce_loss = F.binary_cross_entropy(torch.sigmoid(eval_logits), F.one_hot(eval_target, num_classes=2).to(torch.float32))
                    ema_eval_loss = 0.9*ema_eval_loss + 0.1*eval_bce_loss
                acc = total_acc_account/total_account

                logging.warning(f"eval_ema_loss: {ema_eval_loss.item()}, eval_acc: {acc}")
                model.train()

# step4 测试代码
if __name__ == "__main__":
    if torch.cuda.is_available():
        logging.warning("Cuda is available!")
        if torch.cuda.device_count() > 1:
            logging.warning(f"Find {torch.cuda.device_count()} GPUs !")
            BATCH_SIZE = BATCH_SIZE * torch.cuda.device_count()
        else:
            logging.warning("Too few GPUs!")
            raise Exception("Too few GPUs!")
    else:
        logging.warning("Cuda is not available!")
        raise Exception("Cuda is not available!")
    
    n_gpus = 2
    torch.distributed.init_process_group("nccl", world_size=n_gpus, rank=args.local_rank)
    torch.cuda.device(args.local_rank)

    model = GCNN()
    #  model = TextClassificationModel()
    print("模型总参数:", sum(p.numel() for p in model.parameters()))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    train_data_iter = IMDB(root='.data', split='train') # Dataset类型的对象
    eval_data_iter = IMDB(root='.data', split='test') # Dataset类型的对象
    resume = ""

    train(to_map_style_dataset(train_data_iter), to_map_style_dataset(eval_data_iter), model, optimizer, num_epoch=10, log_step_interval=20, save_step_interval=500, eval_step_interval=300, save_path="./logs_imdb_text_classification", resume=resume)

