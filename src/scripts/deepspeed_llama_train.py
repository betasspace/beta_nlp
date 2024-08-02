import argparse
import math
import time
import os

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data.distributed import DistributedSampler

import deepspeed
import deepspeed.comm as dist
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam
from deepspeed.runtime.utils import set_random_seed

from transformers import get_scheduler
from transformers import default_data_collator
# from utils.data.data_utils import create_pretrain_dataset

parser = argparse.ArgumentParser(description='Bert')
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

# Prepare the data
# create_pretrain_dataset: 一个自定义函数，用于创建预训练数据集。
train_dataset, eval_dataset = create_pretrain_dataset(
    args.local_rank,
    args.data_path,
    args.data_split,
    args.data_output_path,
    args.seed,
    tokenizer,
    args.max_seq_len)
# DataLoaders creation:
# 通过检查 args.local_rank 是否为 −1，代码决定使用普通的采样器(单机)还是分布式采样器 (多机)。
# DistributedSampler 确保在分布式训练环境中，每个进程或节点都能获得数据的一个不重 复的子集，这使得分布式训练变得可能。
# 而在单机环境中，使用常规的随机或顺序采样器即可。
if args.local_rank == -1:
    train_sampler = RandomSampler(train_dataset)
    eval_sampler = SequentialSampler(eval_dataset)
else:
    train_sampler = DistributedSampler(train_dataset)
    eval_sampler = DistributedSampler(eval_dataset)

# default_data_collator: Transformers 库提供的默认数据收集器，用于将多个样本整合为一个批 量数据。
# DataLoader: PyTorch 提供的工具，用于从数据集加载数据到模型进行训练或评估。
train_dataloader = DataLoader(train_dataset,
                              collate_fn=default_data_collator,
                              sampler=train_sampler,
                              batch_size=args.per_device_train_batch_size)
eval_dataloader = DataLoader(eval_dataset,
                             collate_fn=default_data_collator,
                             sampler=eval_sampler,
                             batch_size=args.per_device_eval_batch_size)

# 模型载入
model_name_or_path = ""
from transformers import LlamaForCausalLM, LlamaTokenizer, LlamaConfig
# load_hf_tokenizer will get the correct tokenizer and set padding tokens based on the model family
# 使用 from_pretrained 方法来加载预训练的LLaMA分词器
tokenizer = LlamaTokenizer.from_pretrained(
    model_name_or_path, fast_tokenizer=True)

# 为了确保分词器可以处理各种文本长度，还需要进行了填充设置。
# 如果分词 器还没有指定填充符号，将其设置为 [PAD]，并确定填充行为发生在句子的右侧。
# 此外，为了保证 模型能够正确地处理句子结束和填充，还为模型配置设置了结束符号和填充符号的 ID。
if tokenizer.pad_token is None:
    # assert tokenizer.eos_token is not None
    # tokenizer.add_special_tokens({'pad_token': tokenizer.eos_token})
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.padding_side = 'right'

# 使用 from_pretrained 方法来加载预训练的LLaMA模型和配置
model_config = LlamaConfig.from_pretrained(model_name_or_path)
model = LlamaForCausalLM.from_pretrained(model_name_or_path, config=model_config)

model.config.end_token_id = tokenizer.eos_token_id
model.config.pad_token_id = model.config.eos_token_id

# 为了优化模型在硬件上的性能，还需要调整了模型的词汇表嵌入大小，使其成为 8 的倍数。
model.resize_token_embeddings(int(
    8*
    math.ceil(len(tokenizer) / 8.0))) # make the vocab size multiple of 8




# 一组使用权重 衰减，另一组则不使用。这种参数分组有助于正则化模型，防止过拟合，并允许对特定参数应用不同的学习设置。
def get_optimizer_grouped_parameters(model, weight_decay,
                                     no_decay_name_list=[
                                         "bias", "LayerNorm.weight"
                                     ]):
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in model.named_parameters() 
                if (not any(nd in n 
                            for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p for n, p in model.named_parameters()
                if (any(nd in n for nd in no_decay_name_list) and p.requires_grad)
            ],
            "weight_decay": 0.0,
        },
    ]
    return optimizer_grouped_parameters



# 权重衰减
# Split weights in two groups, one with weight decay and the other not.
optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    model, args.weight_decay, args.learning_rate)

# 根据训练设置(如是否在CPU上进行模型参数卸载)，我们可以选择使用Deep-SpeedCPUAdam 或 FusedAdam 优化器。
# 这两种优化器都是对经典的 Adam 优化器进行优化和改进的版本，为大规模训练提供了高效性能。
AdamOptimizer = DeepSpeedCPUAdam if args.offload else FusedAdam

optimizer = AdamOptimizer(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          betas=(0.9, 0.95))
num_update_steps_per_epoch = math.ceil(
    len(train_dataloader) / args.gradient_accumulation_steps)

# 学习率调度
# 不同于固定的学习率，学习率调度器在训练过程中动态调整学习率。
# 例如，在训练初期快速提高学习率以加速收敛，然后在训练中后期逐渐降低学习率以获得更精细的优化。
# 我们的配置考虑了预热步骤、训练的总步数以及其他关键因素。
lr_scheduler = get_scheduler(
    name=args.lr_scheduler_type,
    optimizer=optimizer,
    num_warmup_steps=args.num_warmup_steps,
    num_training_steps=args.num_train_epochs * num_update_steps_per_epoch,
)


GLOBAL_BATCH_SIZE = 32
MICRO_BATCH_SIZE = 4


def get_train_ds_config(offload,
                        stage=2,
                        enable_hybrid_engine=False, 
                        inference_tp_size=1, 
                        release_inference_cache=False, 
                        pin_parameters=True, 
                        tp_gather_partition_size=8, 
                        max_out_tokens=512, 
                        enable_tensorboard=False, 
                        tb_path="",
                        tb_name=""):
    device = "cpu" if offload else "none"
    # ZeRO优化配置:ZeRO(ZeroRedundancyOptimizer)是DeepSpeed提供的一种优化策略，旨 在减少训练中的冗余并加速模型的训练。
    # 其中的参数，如 offload_param 和 offload_optimizer， 允许用户选择是否将模型参数或优化器状态卸载到 CPU。
    zero_opt_dict = {
        "stage": stage,
        "offload_param": {
            "device": device
        },
        "offload_optimizer": {
            "device": device
        },
        "stage3_param_persistence_threshold": 1e4, 
        "stage3_max_live_parameters": 3e7, 
        "stage3_prefetch_bucket_size": 3e7, 
        "memory_efficient_linear": False
    }
    # GLOBAL_BATCH_SIZE: 定义了全局的批次大小。 这通常是所有 GPU 加起来的总批次大小。
    # MICRO_BATCH_SIZE: 定义了每个 GPU 上的微批次大小。
    # 微批次处理可以帮助大型模型在有限的 GPU 内存中运行，因为每次只加载并处理一小部分数据。
    # 混合精度训练:通过设置 fp16 字段，使得模型可以使用 16 位浮点数进行训练，从而加速训 练过程并减少内存使用。
    # 梯度裁剪:通过 gradient_clipping 字段，我们可以防止训练过程中的梯度爆炸问题。
    # 混合引擎配置:hybrid_engine部分允许用户配置更高级的优化选项，如输出分词的最大数量和推理张量的大小。
    # TensorBoard 配置:使用 DeepSpeed 时，可以通过配置选项直接集成 TensorBoard，从而更方便地跟踪训练过程。
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE, 
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE, 
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
            "enabled": True,
            "loss_scale_window": 100
        },
        "gradient_clipping": 1.0, 
        "prescale_gradients": False, 
        "wall_clock_breakdown": False, 
        "hybrid_engine": {
            "enabled": enable_hybrid_engine,
            "max_out_tokens": max_out_tokens,
            "inference_tp_size": inference_tp_size,
            "release_inference_cache": release_inference_cache,
            "pin_parameters": pin_parameters,
            "tp_gather_partition_size": tp_gather_partition_size,
        },
        "tensorboard": {
            "enabled": enable_tensorboard,
            "output_path": f"{tb_path}/ds_tensorboard_logs/", 
            "job_name": f"{tb_name}_tensorboard"
        }
    }

# get_eval_ds_config:此函数提供了 DeepSpeed 的验证集。
# 与训练配置相比，验 证集配置更为简洁，只需要关注模型推理阶段即可。
def get_eval_ds_config(offload, stage=0):
    device = "cpu" if offload else "none" 
    zero_opt_dict = {
        "stage": stage,
        "stage3_param_persistence_threshold": 1e4,
        "offload_param": {
            "device": device
        },
        "memory_efficient_linear": False
    }
    return {
        "train_batch_size": GLOBAL_BATCH_SIZE, 
        "train_micro_batch_size_per_gpu": MICRO_BATCH_SIZE, 
        "steps_per_print": 10,
        "zero_optimization": zero_opt_dict,
        "fp16": {
         "enabled": True
        },
        "gradient_clipping": 1.0, 
        "prescale_gradients": False, 
        "wall_clock_breakdown": False
    }

# 确定运行的设备:
# 首先，代码检查是否有指定的本地 GPU(通过 args.local_rank)。
# 如果没有指定，程序默认使用 CUDA 设备。否则，它会为进程设置指定的 GPU
if args.local_rank == -1:
    device = torch.device("cuda")
else:
    torch.cuda.set_device(args.local_rank)
    device = torch.device("cuda", args.local_rank)
    # Initializes the distributed backend which will take care of sychronizing nodes/GPUs 
    # torch.distributed.init_process_group(backend='nccl')
    # 初始化分布式后端:在分布式训练中，使用 deepspeed.init_distributed() 函数实现每个进程与其他进程的同步，初始化分布式环境。
    deepspeed.init_distributed()

# 获取当前进程的全局排序:在分布式训练中，使用 torch.distributed.get_rank() 获得每个进程的唯一排序或 ID。
args.global_rank = torch.distributed.get_rank()

# 设置 DeepSpeed 配置:根据用户参数(如是否进行 offload、使用哪个 zero stage 等)，构建构建了一个 DeepSpeed 配置字典，来决定训练设置。
ds_config = get_train_ds_config(offload=args.offload,
                                stage=args.zero_stage,
                                enable_tensorboard=args.enable_tensorboard,
                                tb_path=args.tensorboard_path,
                                tb_name="step1_model")
ds_config['train_micro_batch_size_per_gpu'] = args.per_device_train_batch_size
ds_config['train_batch_size'] = args.per_device_train_batch_size * \
    torch.distributed.get_world_size() * args.gradient_accumulation_steps
# If passed along, set the training seed now.
set_random_seed(args.seed)

# 同步所有工作进程:使用 torch.distributed.barrier() 确保在进一步的初始化之前所有进程都已同步。
torch.distributed.barrier()

# DeepSpeed初始化:
# 这是最关键的一步。
# 通过deepspeed.initialize，可以将模型、优化器、参数和先前构建的 DeepSpeed 配置传递给库，进行初始化。
# 这个函数会返回一个已经根据 DeepSpeed 配置进行了优化的模型和优化器。
model, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    optimizer=optimizer, 
    args=args, 
    config=ds_config, 
    lr_scheduler=lr_scheduler, 
    dist_init_required=True)

# 梯度检查点:对于特别大的模型，梯度检查点是一种节省显存的技巧，即只在需要时计算模型的中间梯度。
# 如果用户启用了这个选项，则会调用 model.gradient_checkpointing_enable() 方法来实现相关功能。
if args.gradient_checkpointing:
    model.gradient_checkpointing_enable()




def print_rank_0(msg, rank=0):
    if rank <= 0:
        print(msg)
# This function can be used to print throughput for Step 1 and 2 only
def print_throughput(hf_model, args, e2e_time, rank=0):
    if rank <= 0:
        hf_config = hf_model.config
        num_layers, hidden_size, vocab_size = get_hf_configs(hf_config)

        gpus_per_model = torch.distributed.get_world_size()
        seq_length = args.max_seq_len
        batch_size = args.per_device_train_batch_size
        samples_per_second = batch_size / e2e_time
        checkpoint_activations_factor = 4 if args.gradient_checkpointing else 3
        if args.lora_dim > 0:
            k = args.lora_dim * 2 / hidden_size
            checkpoint_activations_factor -= (1 - k)

        hf_model._num_params = sum([
            p.ds_numel if hasattr(p, "ds_tensor") else p.numel()
            for p in hf_model.parameters()
        ])
        params_in_billions = hf_model._num_params / (1e9)

        # Megatron paper's formula to calculate training flops
        train_flops_per_iteration = calculate_flops(
            checkpoint_activations_factor, batch_size, seq_length, hf_config)
        
        train_tflops = train_flops_per_iteration / (e2e_time * gpus_per_model *
                                                    (10**12))
        
        param_string = f"{params_in_billions:.3f} B" if params_in_billions != 0 else "NA"
        print(
            f"Model Parameters: {param_string}, Latency: {e2e_time:.2f}s, \
            TFLOPs: {train_tflops:.2f}, Samples/sec: {samples_per_second:.2f}, \
            Time/seq {e2e_time/batch_size:.2f}s, Batch Size: {batch_size}, \
            Sequence Length: {seq_length}"
        )


def save_hf_format(model, tokenizer, args, sub_folder=""):
    # used to save huggingface format, so we can use it for hf.from_pretrained
    model_to_save = model.module if hasattr(model, 'module') else model
    CONFIG_NAME = "config.json"
    WEIGHTS_NAME = "pytorch_model.bin"
    output_dir = os.path.join(args.output_dir, sub_folder)
    os.makedirs(output_dir, exist_ok=True)
    output_model_file = os.path.join(output_dir, WEIGHTS_NAME)
    output_config_file = os.path.join(output_dir, CONFIG_NAME)
    save_dict = model_to_save.state_dict()
    for key in list(save_dict.keys()):
        if "lora" in key:
            del save_dict[key]
    torch.save(save_dict, output_model_file)
    model_to_save.config.to_json_file(output_config_file)
    tokenizer.save_vocabulary(output_dir)


def save_zero_three_model(model_ema, global_rank, save_dir, zero_stage=0):
    zero_stage_3 = (zero_stage == 3)
    os.makedirs(save_dir, exist_ok=True)
    WEIGHTS_NAME = "pytorch_model.bin"

    output_model_file = os.path.join(save_dir, WEIGHTS_NAME)
    model_to_save = model_ema.module if hasattr(model_ema,
                                                'module') else model_ema
    if not zero_stage_3:
        if global_rank == 0:
            torch.save(model_to_save.state_dict(), output_model_file)
    else:
        output_state_dict = {}
        def _z3_params_to_fetch(param_list):
            return [p for p in param_list if hasattr(p, "ds_id") and p.ds_status == deepspeed.zero.ZeroParamStatus.NOT_AVAILABLE]
        for k, v in model_to_save.named_parameters():
            if hasattr(v, 'ds_id'):
                with deepspeed.zero.GatheredParameters(_z3_params_to_fetch([v]),
                                                       enabled=zero_stage_3):
                    v_p = v.data.cpu()
            else:
                v_p = v.cpu()
            if global_rank == 0 and "lora" not in k:
                output_state_dict[k] = v_p 
        if global_rank == 0:
            torch.save(output_state_dict, output_model_file) 
        del output_state_dict

# Train!
# 训练前的准备:使用print_rank_0函数输出当前的训练状态。
# 该函数确保只有指定的进程(通 常是主进程)会打印消息，避免了多进程环境下的重复输出。
# 在开始训练之前，对模型进行 了一次评估，计算模型的困惑度。


print_rank_0("***** Running training *****", args.global_rank)
print_rank_0(
        f"***** Evaluating perplexity, \
        Epoch {0}/{args.num_train_epochs} *****", 
        args.global_rank)

perplexity = evaluation(model, eval_dataloader)
print_rank_0(f"ppl: {perplexity}", args.global_rank)

# 训练循环:每个周期的开始，都会打印当前周期和总周期数。
# 在每次迭代中，数据批次首先被移 动到相应的 GPU 设备，接着模型对这个批次进行前向传播计算损失。
# 使用 model.backward(loss) 计算梯度，并使用 model.step() 更新模型参数。
# 对于主进程，还会使用 print_throughput 函数 打印吞吐量，这有助于了解模型的训练速度和效率。
for epoch in range(args.num_train_epochs):
    print_rank_0(
        f"Beginning of Epoch {epoch+1}/{args.num_train_epochs}, \
        Total Micro Batches {len(train_dataloader)}",
        args.global_rank)
    model.train()
    for step, batch in enumerate(train_dataloader):
        start = time.time()
        batch = to_device(batch, device)
        outputs = model(**batch, use_cache=False)
        loss = outputs.loss
        if args.print_loss:
            print(
                f"Epoch: {epoch}, Step: {step}, \
                Rank: {torch.distributed.get_rank()}, loss = {loss}"
            )
        model.backward(loss)
        model.step()
        end = time.time()
        if torch.distributed.get_rank() == 0:
            print_throughput(model.model, args, end - start,
                             args.global_rank)
if args.output_dir is not None:
    print_rank_0('saving the final model ...', args.global_rank)

# 保存模型:
# 如果指定了输出目录，模型的状态和配置将被保存。
# 模型可以在不同的格式中保 存，例如 Hugging Face 的模型格式或 DeepSpeed 的 Zero Stage 3 特定格式。
# save_hf_format 函数用于保存模型为 Hugging Face 格式，这意味着训练后的模型可以使用 Hugging Face 的 from_pretrained 方法直接加载。
# 对于 Zero Stage 3，save_zero_three_model 函数负责保存，因为在这个阶段，每个 GPU 只保存了模型的一部分。
model = convert_lora_to_linear_layer(model)
if args.global_rank == 0:
    save_hf_format(model, tokenizer, args)

if args.zero_stage == 3:
    # For zero stage 3, each gpu only has a part of the model, so we need a special save function
    save_zero_three_model(model,
                          args.global_rank,
                          args.output_dir,
                          zero_stage=args.zero_stage)




