import os

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

import json
import numpy as np
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer
from torch.utils.data import DataLoader, Dataset
import pandas as pd

model_dir = '/workspace/ckpt/chatglm3-6b-base'
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
base_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
max_length = 1024
key_token1 = 'True'
key_token1 = tokenizer.encode(key_token1)[-1]
key_token2 = 'False'
key_token2 = tokenizer.encode(key_token2)[-1]


class ChatGLM_Filter(nn.Module):
    def __init__(self, base):
        super(ChatGLM_Filter, self).__init__()
        self.base_model = base

    def forward(self, input_ids, attention_mask):
        outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask).logits[:, -1]
        return outputs


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
filter_model = ChatGLM_Filter(base_model)
filter_model.load_state_dict(torch.load("/workspace/ckpt/rm_best_checkpoint_3.pt"))
filter_model.to(device)
filter_model.eval()


def get_orm_score(question, answer):
    with torch.no_grad():
        encoded_pair = tokenizer.encode_plus(
            question + '[answer]' + answer,
            padding='max_length',
            max_length=max_length,  # Set the max length
            truncation=True,
            return_tensors='pt',  # Return PyTorch Tensor format
        )
        input_ids = encoded_pair['input_ids'].cuda()
        attention_mask = encoded_pair['attention_mask'].cuda()
        outputs = filter_model(input_ids, attention_mask)
        outputs = torch.softmax(outputs, dim=1)

        outputs_1 = outputs[0, key_token1].item()
        outputs_2 = outputs[0, key_token2].item()
        score = outputs_1 - outputs_2
        return score


def get_orm_scores(outputs):
    scores = []
    for output in outputs:
        question = output['content']
        answer = output['solution']
        score = get_orm_score(question, answer)
        print(f"Get an ORM score :{score}\n")
        scores.append(score)
    return scores


def get_best_solution_orm(outputs):
    scores = get_orm_scores(outputs)
    best_idx = np.argmax(scores)
    best_output = outputs[best_idx]
    return best_output
