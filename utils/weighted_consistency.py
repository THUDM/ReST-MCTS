from utils.orm_score import get_orm_scores
import re


def get_weighted_consistency_output_scibench(outputs):
    scores = get_orm_scores(outputs)
    output_count = {}
    for i in range(len(outputs)):
        output = outputs[i]
        score = scores[i]
        summ = output['summary'].strip()
        try:
            match = re.findall(r'[^^{.\-0123456789]-?[0-9]+\.?[0-9]*[^^}.0123456789]', summ)[-1][1:][:-1]
            model_ans = float(match)

        except Exception as e:
            try:
                match = re.findall(r'-?[0-9]+\.?[0-9]*', summ)[-1]
                model_ans = float(match)
            except Exception as e:
                print(f'提取答案出错！错误类型:{e}\n')
                continue

        if model_ans not in output_count.keys():
            output_count.update({model_ans: [score, output]})
        else:
            output_count[model_ans][0] += score

    if not output_count:
        return outputs[0]

    most_cons_score = 0
    most_cons_output = {}
    for ans, info in output_count.items():
        if info[0] > most_cons_score:
            most_cons_score = info[0]
            most_cons_output = info[1]
    return most_cons_output
