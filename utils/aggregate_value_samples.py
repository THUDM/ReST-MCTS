import argparse
import os
import pathlib
import json
import pandas as pd
from utils.json_operator import *


def aggregate(arguments):
    base_dir = os.getcwd()
    source_dir = f'{base_dir}/generation/vm/{arguments.task_name}/{arguments.file}/{arguments.mode}'
    if arguments.best_k > 1:
        out_file = f'{base_dir}/generation/vm/{arguments.task_name}/{arguments.file}/{arguments.mode}/{arguments.propose_method}_{arguments.value_method}_best@{arguments.best_k}_all.json'
    else:
        out_file = f'{base_dir}/generation/vm/{arguments.task_name}/{arguments.file}/{arguments.mode}/{arguments.propose_method}_{arguments.value_method}_all.json'

    cur_outputs = read_json(out_file)
    all_outputs = cur_outputs
    if cur_outputs:
        if arguments.best_k > 1:
            df = pd.read_csv(
                f'{base_dir}/generation/vm/{arguments.task_name}/{arguments.file}/{arguments.mode}/{arguments.propose_method}_{arguments.value_method}_best@{arguments.best_k}_all_process.csv')
            process_dict = {col: df[col].iloc[0] for col in df.columns}
            assert sum([value for value in process_dict.values()]) == len(
                cur_outputs), 'process_dict length not match cur_outputs length!\n'
        else:
            df = pd.read_csv(
                f'{base_dir}/generation/vm/{arguments.task_name}/{arguments.file}/{arguments.mode}/{arguments.propose_method}_{arguments.value_method}_all_process.csv')
            process_dict = {col: df[col].iloc[0] for col in df.columns}
            assert sum([value for value in process_dict.values()]) == len(
                cur_outputs), 'process_dict length not match cur_outputs length!\n'
        for file in os.listdir(source_dir):
            if arguments.best_k > 1:
                if f'{arguments.propose_method}_{arguments.value_method}' in file and f'best@{arguments.best_k}' in file and 'all' not in file:
                    this_output = read_json(f'{source_dir}/{file}')
                    this_new_output = this_output[process_dict[file]:]
                    all_outputs.extend(this_new_output)
                    process_dict[file] = len(this_output)
            else:
                if f'{arguments.propose_method}_{arguments.value_method}' in file and 'best@' not in file and 'all' not in file:
                    this_output = read_json(f'{source_dir}/{file}')
                    this_new_output = this_output[process_dict[file]:]
                    all_outputs.extend(this_new_output)
                    process_dict[file] = len(this_output)
        new_df = pd.DataFrame({file: [process_dict[file]] for file in process_dict.keys()})
        if arguments.best_k > 1:
            new_df.to_csv(
                f'{base_dir}/generation/vm/{arguments.task_name}/{arguments.file}/{arguments.mode}/{arguments.propose_method}_{arguments.value_method}_best@{arguments.best_k}_all_process.csv',
                index=False)
        else:
            new_df.to_csv(
                f'{base_dir}/generation/vm/{arguments.task_name}/{arguments.file}/{arguments.mode}/{arguments.propose_method}_{arguments.value_method}_all_process.csv',
                index=False)
    else:
        process_dict = {}
        for file in os.listdir(source_dir):
            if arguments.best_k > 1:
                if f'{arguments.propose_method}_{arguments.value_method}' in file and f'best@{arguments.best_k}' in file and 'all' not in file:
                    this_output = read_json(f'{source_dir}/{file}')
                    all_outputs.extend(this_output)
                    process_dict.update({file: [len(this_output)]})
            else:
                if f'{arguments.propose_method}_{arguments.value_method}' in file and 'best@' not in file and 'all' not in file:
                    this_output = read_json(f'{source_dir}/{file}')
                    all_outputs.extend(this_output)
                    process_dict.update({file: [len(this_output)]})
        new_df = pd.DataFrame(process_dict)
        if arguments.best_k > 1:
            new_df.to_csv(
                f'{base_dir}/generation/vm/{arguments.task_name}/{arguments.file}/{arguments.mode}/{arguments.propose_method}_{arguments.value_method}_best@{arguments.best_k}_all_process.csv',
                index=False)
        else:
            new_df.to_csv(
                f'{base_dir}/generation/vm/{arguments.task_name}/{arguments.file}/{arguments.mode}/{arguments.propose_method}_{arguments.value_method}_all_process.csv',
                index=False)

    dump_json(out_file, all_outputs)


def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--task_name', type=str, default='gsm_8k')
    base_args.add_argument('--file', type=str, default='gsm8k_self_train_1')  # json
    base_args.add_argument('--propose_method', type=str, choices=['gpt', 'glm', 'llama', 'mistral', 'local'], default='local')
    base_args.add_argument('--value_method', type=str, choices=['gpt', 'glm', 'local'], default='local')
    base_args.add_argument('--mode', type=str, choices=['cot', 'tot', 'mcts'], default='mcts')
    base_args.add_argument('--algorithm', type=str, choices=['dfs', 'bfs'], default='dfs')
    base_args.add_argument('--generate_num', type=int, default=1)
    base_args.add_argument('--best_k', type=int, default=1)  # best@k

    arguments = base_args.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    aggregate(args)
