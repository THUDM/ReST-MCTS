import os
import pathlib
from CoT.task import CoT_Task
from ToT.task import ToT_Task
from MCTS.task import MCTS_Task
import argparse
from utils.visualize import visualize
from utils.json_operator import *
from utils.verify_answer import *
from utils.self_consistency import get_consistency_output_scibench


def run(arguments):
    print('-'*30, 'Begin testing', '-'*30, '\n')
    file = f'data/{arguments.task_name}/{arguments.file}.json'
    try:
        data_list = read_json(file)
        data_len = len(data_list)
    except Exception as e:
        print(f'File must be standardized json!\nError type:{e}\n')
        return
    assert data_len > 0, "Data list is empty!\n"
    assert 'content' in data_list[0].keys() and 'answer' in data_list[0].keys(), "Key error, Make sure json object contain correct keys!\n"

    output_list = []
    correct_count = 0
    for i in range(data_len):
        # solve
        print(f'Begin to solve the problem {i+1}...\n')
        data = data_list[i]['content']
        answer = data_list[i]['answer']
        if arguments.mode == 'cot':
            Task = CoT_Task(data, arguments.propose_method, arguments.value_method, arguments.temperature, evaluate=arguments.evaluate)
            if arguments.consistency:
                outputs = []
                for cnt in range(3):
                    output = Task.run()
                    outputs.append(output)
                output = get_consistency_output_scibench(outputs)
            else:
                output = Task.run()

        elif arguments.mode == 'tot':
            Task = ToT_Task(data, arguments.propose_method, arguments.value_method, arguments.algorithm,
                            arguments.branch, arguments.select_branch, arguments.max_depth, arguments.end_gate,
                            arguments.select_method, arguments.temperature, use_case_prompt=arguments.use_case_prompt,
                            low=arguments.low, high=arguments.high, evaluate=arguments.evaluate)
            output, root = Task.run()
            if arguments.visualize:
                visualize(root, Task, arguments.task_name, arguments.file, i + 1)
        else:
            Task = MCTS_Task(data, arguments.propose_method, arguments.value_method, arguments.branch, arguments.end_gate,
                             arguments.roll_policy, arguments.roll_branch, arguments.roll_forward_steps, arguments.time_limit,
                             arguments.iteration_limit, arguments.exploration_constant, arguments.alpha, arguments.inf,
                             arguments.temperature, use_case_prompt=arguments.use_case_prompt, use_reflection=arguments.use_reflection,
                             low=arguments.low, high=arguments.high, evaluate=arguments.evaluate)
            output, root = Task.run()
            if arguments.visualize:
                visualize(root, Task, arguments.task_name, arguments.file, i + 1)

        # evaluate metrics
        if arguments.evaluate:
            result = verify_float(answer, output['summary'])
            output.update({'answer': answer, 'accurate': result})
            if result:
                print(f'The answer of problem {i+1} is correct.\n')
                correct_count += 1
            else:
                print(f'The answer of problem {i+1} is wrong.\n')
        print(f'The solution to problem {i+1} is complete.\n')

        # output
        base_dir = os.getcwd()
        output_dir = pathlib.Path(f'{base_dir}/outputs/{arguments.task_name}/{arguments.file}/{Task.mode}')
        output_file = f'{base_dir}/outputs/{arguments.task_name}/{arguments.file}/{Task.mode}/{Task.propose_method}_{Task.value_method}.json'
        output_list.append(output)
        pathlib.Path.mkdir(output_dir, exist_ok=True, parents=True)
        dump_json(output_file, output_list)

    print('_' * 60)
    # accuracy
    if args.evaluate:
        print(f'Test accuracy:{correct_count / data_len}\n')
        print(f'Correct number of problems:{correct_count}\nTotal number of questions:{data_len}\n')
    print('_' * 60)


def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--task_name', type=str, default='scibench')
    base_args.add_argument('--file', type=str, default='thermo_standardized')  # json
    base_args.add_argument('--propose_method', type=str, choices=['gpt', 'glm', 'llama', 'local'], default='glm')
    base_args.add_argument('--value_method', type=str, choices=['gpt', 'glm', 'local'], default='local')
    base_args.add_argument('--mode', type=str, choices=['cot', 'tot', 'mcts'], default='tot')
    base_args.add_argument('--temperature', type=float, default=0.7)
    base_args.add_argument('--time_limit', type=int, default=None)
    base_args.add_argument('--iteration_limit', type=int, default=100)
    base_args.add_argument('--roll_policy', type=str, choices=['random', 'greedy'], default='greedy')
    base_args.add_argument('--exploration_constant', type=float, default=0.4)
    base_args.add_argument('--roll_forward_steps', type=int, default=2)
    base_args.add_argument('--end_gate', type=float, default=0.9)  # End threshold
    base_args.add_argument('--branch', type=int, default=3)
    base_args.add_argument('--roll_branch', type=int, default=1)
    base_args.add_argument('--inf', type=float, default=0.8)
    base_args.add_argument('--evaluate', type=str, default='scibench')  # Whether to evaluate (empty means no evaluation)
    base_args.add_argument('--alpha', type=float, default=0.5)
    base_args.add_argument('--visualize', type=bool, default=False)  # visualization
    base_args.add_argument('--use_case_prompt', type=bool, default=False)  # Use sample prompts
    base_args.add_argument('--use_reflection', type=str, choices=['simple', 'common'], default='simple')  # Use reflective mode
    base_args.add_argument('--low', type=float, default=0)
    base_args.add_argument('--high', type=float, default=1)
    base_args.add_argument('--algorithm', type=str, choices=['dfs', 'bfs'], default='dfs')
    base_args.add_argument('--select_branch', type=int, default=2)
    base_args.add_argument('--max_depth', type=int, default=8)
    base_args.add_argument('--select_method', type=str, choices=['greedy', 'sample'], default='greedy')
    base_args.add_argument('--consistency', type=bool, default=True)

    arguments = base_args.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()
    run(args)
