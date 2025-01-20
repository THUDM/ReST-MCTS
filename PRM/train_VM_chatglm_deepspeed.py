# import debugpy; debugpy.connect(('100.98.26.69', 5690))
import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES']='0,1,2,3' 
import json
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, AdamW
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import time
import deepspeed
import torch.distributed as dist

parser = argparse.ArgumentParser()
parser.add_argument('--debug', action='store_true', help='Enable debug mode')
parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')
args = parser.parse_args()

if 'LOCAL_RANK' in os.environ:
    args.local_rank = int(os.environ['LOCAL_RANK'])
    print('>>>args.local_rank=', args.local_rank)


deepspeed.init_distributed()

max_length = 1024

# Load the pre-trained ChatGLM3-6b model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("/data/llms/chatglm3-6b", trust_remote_code=True)
base_model = AutoModel.from_pretrained("/data/llms/chatglm3-6b",
                                       trust_remote_code=True).bfloat16()

# Custom Dataset class
class MyDataset(Dataset):
    def __init__(self, data_js, tokenizer):
        self.data_js = data_js
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data_js)

    def __getitem__(self, idx):
        prompt_answer = self.data_js[idx]['prompt_answer']
        label = self.data_js[idx]['label']

        encoded_pair = self.tokenizer.encode_plus(
            prompt_answer,
            padding='max_length',
            max_length=max_length,  # Set the max length
            truncation=True,
            return_tensors='pt',  # Return PyTorch Tensor format
        )

        return {
            'input_ids': encoded_pair['input_ids'].squeeze(),
            'attention_mask': encoded_pair['attention_mask'].squeeze(),
            'label': label
        }


class ChatGLM_VM(nn.Module):
    def __init__(self, base, vocab_size, num_classes=1):
        super(ChatGLM_VM, self).__init__()
        self.base_model = base
        self.LN = nn.Linear(vocab_size, num_classes, dtype=torch.bfloat16)

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1]
        value_outputs = self.LN(outputs)
        return value_outputs.squeeze(dim=1)


# Load training set, validation set, and test set data
train_js = '/data/ReST-MCTS-PRM-0th/train_en.json'
test_js = '/data/ReST-MCTS-PRM-0th/test_en.json'
val_js = '/data/ReST-MCTS-PRM-0th/valid_en.json'


def read_json(source):
    json_list = []
    with open(source, 'r', encoding='utf-8') as f:
        for line in f:
            json_list.append(json.loads(line))
    return json_list


train_json = read_json(train_js)  # This section uses a CSV file as an example to describe how to load data
val_json = read_json(val_js)
test_json = read_json(test_js)

if args.debug:
    print(">>>Debug mode: Using only toy training/val/test samples")
    train_json = train_json[:2]
    val_json = val_json[:2] 
    test_json = test_json[:2] 

# Create a custom dataset
train_dataset = MyDataset(train_json, tokenizer)
val_dataset = MyDataset(val_json, tokenizer)
test_dataset = MyDataset(test_json, tokenizer)

# Create data loaders
batch_size = 2  # 3  # Set batch size
gradient_accumulation_steps = 4

train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

# Set device and model
device = torch.device("cuda", args.local_rank)
print(device, '\n')
vocab_size = base_model.config.padded_vocab_size
print(vocab_size)
VM = ChatGLM_VM(base_model, vocab_size, 1)


ds_config = {
    "train_micro_batch_size_per_gpu": batch_size,  
    "gradient_accumulation_steps": gradient_accumulation_steps, 
    "train_batch_size": batch_size * torch.cuda.device_count() * gradient_accumulation_steps, 
    
    "bf16": {
        "enabled": True 
    },

    "zero_optimization": {
        "stage": 1,  
        "allgather_partitions": True,
        "reduce_scatter": True
    },
    # "zero_optimization": {
    #     "stage": 0,
    # },  
    "zero_allow_untested_optimizer": True,
    
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 2e-5
        }
    }
}




model_engine, optimizer, _, _ = deepspeed.initialize(
    args=args,
    model=VM,
    model_parameters=VM.parameters(),
    config=ds_config
)

# Define loss function
criterion = nn.MSELoss()
num_epochs = 3 # or 3 
# Training and validation loop
best_val_loss = float('inf')
train_losses = []
val_losses = []

train_start_time = time.time()
for epoch in range(num_epochs):
    if args.local_rank == 0:
        print(f"{epoch}/{num_epochs} training")
    # Training
    model_engine.train()
    train_loss = 0.0
    train_sampler.set_epoch(epoch)
    
    for batch in tqdm(train_dataloader, disable=args.local_rank != 0):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].bfloat16().to(device)

        outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        
        model_engine.backward(loss)
        model_engine.step()

        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_dataloader)
    train_losses.append(avg_train_loss)

    # Validation
    if args.local_rank == 0:
        model_engine.eval()
        val_loss = 0.0
        val_labels = []
        with torch.no_grad():
            for batch in tqdm(val_dataloader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['label'].bfloat16().to(device)
                outputs = model_engine(input_ids=input_ids, attention_mask=attention_mask)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                val_labels.extend(labels.tolist())

        avg_val_loss = val_loss / len(val_dataloader)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch + 1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} ")

        # Save best model
        if avg_val_loss < best_val_loss:
            print(">>>Save best model...")
            best_val_loss = avg_val_loss
            
            # model_engine.save_checkpoint("/data/records/Chatglm", tag="VM_best_checkpoint")
            model_engine.save_16bit_model(
                "/data/records/Chatglm", 
                "VM_best_checkpoint_0117.pt"
            )

if args.local_rank == 0:
    train_end_time = time.time()
    print("PRM Training complete!")
    print(f"PRM Training time: {train_end_time - train_start_time:.2f} seconds")

