import os
import json
import re
import pathlib
import argparse
from utils.json_operator import *
from MCTS.task import *
from tqdm import tqdm


def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--task_name', type=str, default='gsm_8k')
    base_args.add_argument('--file', type=str, default='gsm8k_all')  # json
    base_args.add_argument('--propose_method', type=str, choices=['llama', 'mistral', 'local'],
                           default='mistral')

    arguments = base_args.parse_args()
    return arguments


def do_vm_critic_mc(arguments):
    base_dir = os.getcwd()
    policy_output_file = f'{base_dir}/extracted_samples/{arguments.task_name}/{arguments.file}/mcts/policy_samples/{arguments.propose_method}_local.json'
    datas = read_json(policy_output_file)
    out_file = f'{base_dir}/extracted_samples/{arguments.task_name}/{arguments.file}/mcts/policy_samples/{arguments.propose_method}_local_vm_critic.json'
    done_datas = read_json(out_file)
    done_idx = len(done_datas)
    new_datas = done_datas
    idx = 0
    for data in tqdm(datas):
        if idx < done_idx:
            idx += 1
            continue
        question = data['content']
        solution = data['summary']
        task = MCTS_Task(question, propose_method=arguments.propose_method, value_method='local', lang='en')

        score = None
        cnt = 3
        while score is None and cnt:
            score = task.get_step_value(solution)
            cnt -= 1
        if score is None:
            score = 0
        data.update({'vm_critic': score})
        new_datas.append(data)
        dump_json(out_file, new_datas)
        idx += 1


if __name__ == '__main__':
    args = parse_args()
    do_vm_critic_mc(args)
