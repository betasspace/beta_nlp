import argparse
import os
import shutil
import time
import warnings
import numpy as np
warnings.filterwarnings('ignore')
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from torch.utils.data.distributed import DistributedSampler
from models import DeepLab
from dataset import Cityscaples


parser = argparse.ArgumentParser(description='DeepLab')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=3, type=int,
                    metavar='N')
parser.add_argument('--local_rank', default=0, type=int, help='node rank for distributed training')
args = parser.parse_args()
torch.distributed.init_process_group(backend="nccl") # 初始化 print("Use GPU: {} for training".format(args.local_rank))
# create model
model = DeepLab()
torch.cuda.set_device(args.local_rank) # 当前显卡 model = model.cuda() # 模型放在显卡上

model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True) # 数据并行
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(model.parameters(), args.lr,
        momentum=args.momentum, weight_decay=args.weight_decay)
train_dataset = Cityscaples()
train_sampler = DistributedSampler(train_dataset) # 分配数据
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True, sampler=train_sampler)

# comment: 
# CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 main.py


