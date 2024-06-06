from utils.json_operator import *
from CoT.task import *


def evaluate_result(source_dir: str, result_file_pattern: str):
    result_files = []
    for file in os.listdir(source_dir):
        if result_file_pattern in file:
            result_files.append(file)

    all_results = []
    for result_file in result_files:
        result_file_path = os.path.join(source_dir, result_file)
        result = read_json(result_file_path)
        all_results.extend(result)

    total_num = len(all_results)
    simulate_corr_count = 0
    for single_result in all_results:
        corr_num = single_result['correct_num']
        sample_num = single_result['sample_num']
        acc = corr_num / sample_num
        simulate_corr_count += acc

    simulate_acc = simulate_corr_count / total_num
    print(f'simulate_acc: {simulate_acc}, corr_expectation: {simulate_corr_count}, total_num: {total_num}')


def reEvaluate_result(source_dir: str, result_file_pattern: str, multisample=False):
    result_files = []
    for file in os.listdir(source_dir):
        if result_file_pattern in file:
            result_files.append(file)

    all_results = []
    for result_file in result_files:
        result_file_path = os.path.join(source_dir, result_file)
        result = read_json(result_file_path)
        all_results.extend(result)

    total_num = len(all_results)
    if multisample:
        simulate_corr_count = 0
        for results in all_results:
            single_corr_num = 0
            answer = results['real_answer']
            question = results['content']
            for single_result in results['samples']:
                if single_result['accurate']:
                    single_corr_num += 1
                else:
                    if exact_match_score(single_result['summary'], answer):
                        single_corr_num += 1
                        single_result['accurate'] = True
                    else:
                        solution = single_result['solution'].strip()
                        Task = CoT_Task(question, propose_method='local', value_method='local', evaluate='math',
                                        lang='en', answer=answer)
                        cnt = 10
                        summary = ''
                        while not solution and cnt:
                            out = Task.run()
                            solution = out['solution']
                            summary = out['summary']
                            cnt -= 1
                        single_result['solution'] = solution

                        if '####' in solution:
                            summary = 'The final answer is ' + solution.split('####')[-1].strip()
                        elif 'The final answer is' in solution:
                            summary = 'The final answer is ' + solution.split('The final answer is')[-1].strip()
                        elif 'The answer is' in solution:
                            summary = 'The final answer is ' + solution.split('The answer is')[-1].strip()
                        else:
                            cnt = 10
                            while cnt and not summary:
                                summary = Task.get_MATH_summary(solution)
                                cnt -= 1

                        result = exact_match_score(summary, answer)
                        if result:
                            single_corr_num += 1
                            single_result['accurate'] = True
                        single_result['summary'] = summary

            sample_num = results['sample_num']
            results['correct_num'] = single_corr_num
            acc = single_corr_num / sample_num
            simulate_corr_count += acc

        simulate_acc = simulate_corr_count / total_num
        print(f'simulate_acc: {simulate_acc}, corr_expectation: {simulate_corr_count}, total_num: {total_num}')
        return all_results

    else:
        simulate_corr_count = 0
        for results in all_results:
            answer = results['real_answer']
            question = results['content']
            solution = results['solution']
            if results['accurate']:
                simulate_corr_count += 1
            else:
                if exact_match_score(results['summary'], answer):
                    simulate_corr_count += 1
                    results['accurate'] = True
                else:
                    Task = CoT_Task(question, propose_method='local', value_method='local', evaluate='math',
                                    lang='en', answer=answer)
                    cnt = 10
                    summary = ''
                    while not solution and cnt:
                        out = Task.run()
                        solution = out['solution']
                        summary = out['summary']
                        cnt -= 1
                    results['solution'] = solution

                    if '####' in solution:
                        summary = 'The final answer is ' + solution.split('####')[-1].strip()
                    elif 'The final answer is' in solution:
                        summary = 'The final answer is ' + solution.split('The final answer is')[-1].strip()
                    elif 'The answer is' in solution:
                        summary = 'The final answer is ' + solution.split('The answer is')[-1].strip()
                    else:
                        cnt = 10
                        while cnt and not summary:
                            summary = Task.get_MATH_summary(solution)
                            cnt -= 1

                    result = exact_match_score(summary, answer)
                    if result:
                        simulate_corr_count += 1
                        results['accurate'] = True
                    results['summary'] = summary

        simulate_acc = simulate_corr_count / total_num
        print(f'simulate_acc: {simulate_acc}, corr_expectation: {simulate_corr_count}, total_num: {total_num}')
        return all_results
