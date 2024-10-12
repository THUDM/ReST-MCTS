import os
os.environ['CUDA_VISIBLE_DEVICES'] = '6'
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
    base_args.add_argument('--propose_method', type=str, choices=['gpt', 'glm', 'llama', 'mistral', 'local'],
                           default='mistral')
    base_args.add_argument('--start_num', type=int, default=225)  # Starting sequence number (not absolute sequence number)
    base_args.add_argument('--end_num', type=int, default=450)
    base_args.add_argument('--generate_num', type=int, default=256)
    base_args.add_argument('--do_aggregate', type=bool, default=False)  # aggregate results

    arguments = base_args.parse_args()
    return arguments


def do_vm_critic(arguments):
    base_dir = os.getcwd()
    policy_output_file = f'{base_dir}/generation/vm/{arguments.task_name}/{arguments.file}/cot/{arguments.propose_method}_local_all.json'
    datas = read_json(policy_output_file)
    assert len(datas) % arguments.generate_num == 0, 'length not match!\n'
    out_file = f'{base_dir}/generation/vm/{arguments.task_name}/{arguments.file}/cot/{arguments.propose_method}_local_vm_critic_{arguments.start_num}_{arguments.end_num}.json'
    done_datas = read_json(out_file)
    done_idx = len(done_datas)
    new_datas = done_datas
    idx = 0
    for data in tqdm(datas):
        if idx < arguments.start_num * arguments.generate_num + done_idx or idx >= arguments.end_num * arguments.generate_num:
            idx += 1
            continue
        question = data['content']
        solution = data['solution'] + '\n' + data['summary']
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


def aggregate_vm_critic(arguments):
    base_dir = os.getcwd()
    out_dir = f'{base_dir}/generation/vm/{arguments.task_name}/{arguments.file}/cot'
    pattern = f'{arguments.propose_method}_local_vm_critic'
    all_outputs = []
    for file in os.listdir(out_dir):
        if pattern in file and 'all' not in file:
            all_outputs.extend(read_json(f'{out_dir}/{file}'))
    dump_json(f'{out_dir}/{pattern}_all.json', all_outputs)


if __name__ == '__main__':
    args = parse_args()
    if args.do_aggregate:
        aggregate_vm_critic(args)
    else:
        do_vm_critic(args)
