import os
from contextlib import nullcontext
from trl.trainer.utils import DPODataCollatorWithPadding
from utils.json_operator import *
from trl import DPOTrainer, TrlParser, ModelConfig
from transformers import TrainingArguments, AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset
from trl.commands.cli_utils import DpoScriptArguments
from trl.trainer import ppo_config
# import wandb
# wandb.login(key='0')

# model
model_dir = "/workspace/ckpt/MetaMath-Mistral-7B"
model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
model_ref = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=torch.bfloat16)
tokenizer = AutoTokenizer.from_pretrained(model_dir)
model_config = model.config

# data
# d_path = "extracted_samples/self_train/cot/llama_local_critic_dpo.json"
d_path = "extracted_samples/self_train/cot/mistral_local_critic_dpo.json"
data_dict = read_json(d_path)[0]
d_len = len(data_dict['prompt'])
assert d_len == len(data_dict['chosen']) and d_len == len(data_dict['rejected'])
print("data_len:", d_len)
if 'llama' in d_path:
    chat_format = '<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'
    ans_format = '{solution}'
elif 'mistral' in d_path:
    chat_format = '[INST]{query}[/INST]'
    ans_format = '{solution}'
else:
    raise NotImplementedError


def preprocess(row):
    processed_prompt = chat_format.format(query=row['prompt'])
    processed_chosen = ans_format.format(solution=row['chosen'])
    processed_rejected = ans_format.format(solution=row['rejected'])
    processed_example = {
        "prompt": processed_prompt,
        "chosen": processed_chosen,
        "rejected": processed_rejected,
    }
    return processed_example


dataset = Dataset.from_dict(data_dict)
dataset = dataset.map(preprocess, batched=False)
dataset = dataset.train_test_split(test_size=0.05)
train_dataset = dataset['train']
test_dataset = dataset['test']


if __name__ == "__main__":
    ################
    # Training Args
    ################
    args = TrainingArguments(
        output_dir="",
        overwrite_output_dir=True,
        do_train=True,
        do_eval=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        num_train_epochs=2,
        learning_rate=3e-6,
        per_device_train_batch_size=1,
        optim="adamw_torch",
        bf16_full_eval=True,
        bf16=True,
        gradient_accumulation_steps=2,
        per_gpu_eval_batch_size=1,
        remove_unused_columns=False,
        # deepspeed="config/deepspeed_zero3.json"
    )
    ################
    # Tokenizer
    ################
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if tokenizer.chat_template is None:
        tokenizer.chat_template = "{% for message in messages %}{{message['role'] + ': ' + message['content'] + '\n\n'}}{% endfor %}{{ eos_token }}"

    ################
    # Training
    ################
    pad_id = 128001 if 'llama' in d_path else 32000
    collator = DPODataCollatorWithPadding(pad_token_id=pad_id, is_encoder_decoder=model_config.is_encoder_decoder)
    trainer = DPOTrainer(
        model,
        model_ref,
        args=args,
        data_collator=collator,
        dataset_num_proc=8,
        max_length=1024,
        max_prompt_length=256,
        max_target_length=1024,
        train_dataset=train_dataset,
        eval_dataset=test_dataset,
        tokenizer=tokenizer,
        truncation_mode='keep_end',
        beta=0.1,
    )

    # print('train num: ', len(trainer.train_dataset))
    trainer.train()
    trainer.save_model(args.output_dir)
