import os
import re
import pathlib
import argparse
from utils.json_operator import *


def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--task_name', type=str, default='gsm_8k')
    base_args.add_argument('--file', type=str, default='gsm8k_all')  # json
    base_args.add_argument('--propose_method', type=list[str], default=['mistral', 'llama', 'local'])
    base_args.add_argument('--value_threshold', type=float, default=0.2)  # value low gate
    base_args.add_argument('--len_threshold', type=int, default=50)  # str len

    arguments = base_args.parse_args()
    return arguments


def do_filter_policy_examples_by_value(arguments):
    source_dir = f"extracted_samples/{arguments.task_name}/{arguments.file}/mcts/policy_samples"
    for file in os.listdir(source_dir):
        if file.split('_')[0] in arguments.propose_method and 'vm_critic' in file:
            source_file = os.path.join(source_dir, file)
            datas = read_json(source_file)
            selected_datas = []
            for data in datas:
                if data['vm_critic'] >= arguments.value_threshold and len(data['summary']) >= arguments.len_threshold:
                    selected_datas.append(data)
            backend = file.split('_')[0]
            out_file = f'{source_dir}/{backend}_local_vm_filtered.json'
            dump_json(out_file, selected_datas)


if __name__ == '__main__':
    args = parse_args()
    do_filter_policy_examples_by_value(args)
    