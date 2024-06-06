import os
import pathlib
from CoT.task import CoT_Task
from ToT.task import ToT_Task
from MCTS.task import MCTS_Task
import argparse
from utils.visualize import visualize
from utils.json_operator import *


def run(arguments):
    print('-'*30, '开始生成', '-'*30, '\n')
    base_dir = os.getcwd()
    file = f'{base_dir}/data/{arguments.task_name}/{arguments.file}.json'
    print(f'Reading data from {file}...\n')
    try:
        data_list = read_json(file)
        data_len = len(data_list)
    except Exception as e:
        print(f'File must be standardized json!\nError type:{e}\n')
        return
    assert data_len > 0, "Data list is empty!\n"
    assert 'content' in data_list[0].keys() and 'answer' in data_list[0].keys(), "Key error, Make sure json object contain correct keys!\n"

    output_list = []
    process = 0
    if arguments.partial:
        base_dir = os.getcwd()
        output_file = f'{base_dir}/generation/{arguments.reward_type}/{arguments.task_name}/{arguments.file}/{arguments.mode}/{arguments.propose_method}_{arguments.value_method}_start{arguments.start_num}_end{arguments.end_num}.json'
        output_list = read_json(output_file)
        process = len(output_list) / arguments.generate_num

    for i in range(data_len):
        if i < arguments.start_num + process or i >= arguments.end_num:
            continue
        # solve
        print(f'开始解答第{i+1}题...\n')
        data = data_list[i]['content']
        answer = data_list[i]['answer']
        outputs = []

        if arguments.mode == 'mcts':
            Task = MCTS_Task(data, arguments.propose_method, arguments.value_method, arguments.branch, arguments.end_gate,
                             arguments.roll_policy, arguments.roll_branch, arguments.roll_forward_steps, arguments.time_limit,
                             arguments.iteration_limit, arguments.exploration_constant, arguments.alpha, arguments.inf,
                             arguments.temperature, use_case_prompt=arguments.use_case_prompt, use_reflection=arguments.use_reflection,
                             low=arguments.low, high=arguments.high, evaluate=arguments.evaluate, sample_value='full', answer=answer, verify_method='string', lang=arguments.lang)
            for cnt in range(arguments.generate_num):
                output, root = Task.run()
                corr_policy_sample_num = sum([sample['correct'] for sample in output['policy_samples']])
                total_policy_sample_num = len(output['policy_samples'])
                output.update({'corr_policy_sample_num': corr_policy_sample_num, 'total_policy_sample_num': total_policy_sample_num})
                outputs.append(output)

        elif arguments.mode == 'cot':
            Task = CoT_Task(data, arguments.propose_method, arguments.value_method, arguments.temperature, evaluate=arguments.evaluate, lang=arguments.lang, answer=answer, verify_method='string', do_self_critic=arguments.do_self_critic)
            for cnt in range(arguments.generate_num):
                output = Task.run()
                outputs.append(output)

        else:
            print("Unsupported sample generation mode!\n")
            return

        for output in outputs:
            output_list.append(output)

        print(f'第{i+1}题解答结束。\n')

        # output
        base_dir = os.getcwd()
        output_dir = pathlib.Path(f'{base_dir}/generation/{arguments.reward_type}/{arguments.task_name}/{arguments.file}/{Task.mode}')
        output_file = f'{base_dir}/generation/{arguments.reward_type}/{arguments.task_name}/{arguments.file}/{Task.mode}/{Task.propose_method}_{Task.value_method}_start{arguments.start_num}_end{arguments.end_num}.json'

        pathlib.Path.mkdir(output_dir, exist_ok=True, parents=True)
        dump_json(output_file, output_list)
        print('_' * 60)


def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--task_name', type=str, default='gsm_8k')
    base_args.add_argument('--file', type=str, default='gsm8k_self_train_1')  # json
    base_args.add_argument('--reward_type', type=str, choices=['vm', 'prm'], default='vm')
    base_args.add_argument('--propose_method', type=str, choices=['gpt', 'glm', 'llama', 'mistral', 'local'], default='local')
    base_args.add_argument('--value_method', type=str, choices=['gpt', 'glm', 'local'], default='local')
    base_args.add_argument('--mode', type=str, choices=['cot', 'tot', 'mcts'], default='mcts')
    base_args.add_argument('--temperature', type=float, default=0.7)
    base_args.add_argument('--time_limit', type=int, default=None)
    base_args.add_argument('--iteration_limit', type=int, default=40)
    base_args.add_argument('--roll_policy', type=str, choices=['random', 'greedy'], default='greedy')
    base_args.add_argument('--exploration_constant', type=float, default=0.5)
    base_args.add_argument('--roll_forward_steps', type=int, default=1)
    base_args.add_argument('--end_gate', type=float, default=0.8)
    base_args.add_argument('--branch', type=int, default=3)
    base_args.add_argument('--roll_branch', type=int, default=1)
    base_args.add_argument('--inf', type=float, default=0.9)
    base_args.add_argument('--evaluate', type=str, default='')
    base_args.add_argument('--alpha', type=float, default=0.5)
    base_args.add_argument('--visualize', type=bool, default=False)
    base_args.add_argument('--use_case_prompt', type=bool, default=False)
    base_args.add_argument('--use_reflection', type=str, choices=['simple', 'common'], default='simple')
    base_args.add_argument('--low', type=float, default=0)
    base_args.add_argument('--high', type=float, default=1)
    base_args.add_argument('--algorithm', type=str, choices=['dfs', 'bfs'], default='dfs')
    base_args.add_argument('--select_branch', type=int, default=2)
    base_args.add_argument('--max_depth', type=int, default=8)
    base_args.add_argument('--select_method', type=str, choices=['greedy', 'sample'], default='greedy')
    base_args.add_argument('--generate_num', type=int, default=1)
    base_args.add_argument('--start_num', type=int, default=0)
    base_args.add_argument('--end_num', type=int, default=165)
    base_args.add_argument('--partial', type=bool, default=False)
    base_args.add_argument('--lang', type=str, choices=['zh', 'en'], default='en')
    base_args.add_argument('--do_self_critic', type=bool, default=True)  # for CoT

    arguments = base_args.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    run(args)
