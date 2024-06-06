import json
import re
import os
import pathlib
import argparse
from utils.json_operator import *
from utils.verify_MATH import extract_answer


def parse_args():
    base_args = argparse.ArgumentParser()
    base_args.add_argument('--task_name', type=str, default='gsm_8k')
    base_args.add_argument('--file', type=str, default='gsm8k_all')  # json
    base_args.add_argument('--propose_method', type=str, choices=['gpt', 'glm', 'llama', 'mistral', 'local'],
                           default='mistral')
    base_args.add_argument('--generate_num', type=int, default=256)
    base_args.add_argument('--evaluate_method', type=str, choices=['best', 'weighted'], default='best')
    arguments = base_args.parse_args()
    return arguments


def eval_vm(arguments):
    base_dir = os.getcwd()
    out_file = f'{base_dir}/generation/vm/{arguments.task_name}/{arguments.file}/cot/{arguments.propose_method}_local_vm_critic_all.json'
    datas = read_json(out_file)
    idx = 0
    corr_num = 0
    total_num = 0
    while idx < len(datas):
        total_num += 1
        cur_datas = datas[idx:idx + arguments.generate_num]
        idx += arguments.generate_num
        if arguments.evaluate_method == 'best':
            sorted_cur_datas = sorted(cur_datas, key=lambda x: x['vm_critic'], reverse=True)
            i = 0
            while not sorted_cur_datas[i]['summary'] and i < len(sorted_cur_datas) - 1:
                i += 1
            selected_data = sorted_cur_datas[i]
            if selected_data['accurate']:
                corr_num += 1

        elif arguments.evaluate_method == 'weighted':
            all_answers = {}  # {answer: [idx, summ, value]}
            for i, data in enumerate(cur_datas):
                summ = data['summary']
                if not summ:
                    continue

                extracted_answer = extract_answer(summ)
                if extracted_answer in all_answers.keys():
                    all_answers[extracted_answer][2] += data['vm_critic']
                else:
                    all_answers[extracted_answer] = [i, summ, data['vm_critic']]

            if not all_answers:
                continue
            best_answer = max(all_answers.values(), key=lambda x: x[2])
            best_id = best_answer[0]
            if cur_datas[best_id]['accurate']:
                corr_num += 1

        else:
            print('evaluate_method not implemented')
            raise NotImplementedError

    print(f'测试准确率:{corr_num / total_num}')
    print(f'测试总样本数:{total_num}')
    print(f'测试正确样本数:{corr_num}')


if __name__ == '__main__':
    args = parse_args()
    eval_vm(args)
