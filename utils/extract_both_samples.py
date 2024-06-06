import json
import re
import os
import pathlib
import argparse
from utils.json_operator import *


def extract_both_samples(arguments):
    base_dir = os.getcwd()
    out_file = f'{base_dir}/generation/vm/{arguments.task_name}/{arguments.file}/{arguments.mode}/{arguments.propose_method}_{arguments.value_method}_all.json'
    all_outputs = read_json(out_file)

    if arguments.mode == 'mcts':
        policy_output_dir = pathlib.Path(f'{base_dir}/extracted_samples/{arguments.task_name}/{arguments.file}/{arguments.mode}/policy_samples')
        pathlib.Path.mkdir(policy_output_dir, exist_ok=True, parents=True)
        policy_output_file = f'{base_dir}/extracted_samples/{arguments.task_name}/{arguments.file}/{arguments.mode}/policy_samples/{arguments.propose_method}_{arguments.value_method}.json'
        new_policy_outputs = []
        for output in all_outputs:
            cur_outputs = []
            content = output['content']
            real_answer = output['real_answer']
            for cur_output in output['policy_samples']:
                new_sample = {'content': content, 'summary': cur_output['solution'] + cur_output['summary'] if 'final answer is' not in cur_output['solution'] else cur_output['solution'], 'label': cur_output['correct'], 'real_answer': real_answer}
                cur_outputs.append(new_sample)
            new_policy_outputs.extend(cur_outputs)
        dump_json(policy_output_file, new_policy_outputs)

        value_output_dir = pathlib.Path(f'{base_dir}/extracted_samples/{arguments.task_name}/{arguments.file}/{arguments.mode}/value_samples')
        pathlib.Path.mkdir(value_output_dir, exist_ok=True, parents=True)
        value_output_file = f'{base_dir}/extracted_samples/{arguments.task_name}/{arguments.file}/{arguments.mode}/value_samples/{arguments.propose_method}_{arguments.value_method}.json'
        new_value_outputs = []
        for output in all_outputs:
            cur_outputs = []
            content = output['content']
            for cur_output in output['value_samples']:
                new_sample = {'prompt_answer': 'Problem:' + content + '\nSolution:\n' + cur_output['steps'], 'label': cur_output['value']}
                cur_outputs.append(new_sample)
            new_value_outputs.extend(cur_outputs)
        dump_json(value_output_file, new_value_outputs)

    elif arguments.mode == 'cot':
        policy_output_dir = pathlib.Path(f'{base_dir}/extracted_samples/{arguments.task_name}/{arguments.file}/{arguments.mode}/policy_samples')
        pathlib.Path.mkdir(policy_output_dir, exist_ok=True, parents=True)
        policy_output_file = f'{base_dir}/extracted_samples/{arguments.task_name}/{arguments.file}/{arguments.mode}/policy_samples/{arguments.propose_method}_{arguments.value_method}.json'
        new_policy_outputs = []
        for output in all_outputs:
            content = output['content']
            real_answer = output['real_answer']
            new_sample = {'content': content, 'summary': output['solution'] + output['summary'] if 'final answer is' not in output['solution'] else output['solution'], 'label': output['accurate'], 'real_answer': real_answer}
            if arguments.do_self_critic:
                new_sample.update({'self_critic': output['self_critic']})
            new_policy_outputs.append(new_sample)
        dump_json(policy_output_file, new_policy_outputs)

    else:
        print("Unsupported sample extraction mode!\n")
        return


def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--task_name', type=str, default='gsm_8k')
    base_args.add_argument('--file', type=str, default='gsm8k_self_train_1')  # json
    base_args.add_argument('--propose_method', type=str, choices=['gpt', 'glm', 'llama', 'mistral', 'local'], default='local')
    base_args.add_argument('--value_method', type=str, choices=['gpt', 'glm', 'local'], default='local')
    base_args.add_argument('--mode', type=str, choices=['cot', 'mcts'], default='mcts')
    base_args.add_argument('--generate_num', type=int, default=1)
    base_args.add_argument('--do_self_critic', type=bool, default=False)  # for CoT

    arguments = base_args.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    extract_both_samples(args)
