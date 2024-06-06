import json
import re
import os
import pathlib
import argparse
from utils.json_operator import *
from CoT.task import *
from tqdm import tqdm


def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--task_name', type=str, default='gsm_8k')
    base_args.add_argument('--file', type=str, default='gsm8k_self_train_1')  # json
    base_args.add_argument('--propose_method', type=str, choices=['gpt', 'glm', 'llama', 'mistral', 'local'],
                           default='local')

    arguments = base_args.parse_args()
    return arguments


def do_self_critic(arguments):
    base_dir = os.getcwd()
    policy_output_file = f'{base_dir}/extracted_samples/{arguments.task_name}/{arguments.file}/cot/policy_samples/{arguments.propose_method}_local.json'
    datas = read_json(policy_output_file)
    out_file = f'{base_dir}/extracted_samples/{arguments.task_name}/{arguments.file}/cot/policy_samples/{arguments.propose_method}_local_critic.json'
    new_datas = []
    for data in tqdm(datas):
        question = data['content']
        solution = data['summary']
        task = CoT_Task(question, propose_method=arguments.propose_method, value_method='local', do_self_critic=True)

        score = None
        cnt = 3
        while score is None and cnt:
            score = task.get_self_critic(solution)
            cnt -= 1
        if score is None:
            score = 0
        data.update({'self_critic': score})
        new_datas.append(data)
        dump_json(out_file, new_datas)


if __name__ == '__main__':
    args = parse_args()
    do_self_critic(args)
