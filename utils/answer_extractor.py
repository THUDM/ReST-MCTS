import re

choices = ['A', 'B', 'C', 'D', 'a', 'b', 'c', 'd']


def extract(answer, q_type):
    if '\\boxed{' in answer:
        trunc_ans = answer.split('\\boxed{')[-1]
        extracted_ans = trunc_ans.split('}')[0].strip().replace(' ', '').replace(',', '')
        flag = 1
        if q_type == 'MCQ':
            if len(extracted_ans) == 1 and extracted_ans in choices:
                flag = 1
                extracted_ans = extracted_ans.upper()
            else:
                flag = 0
        elif q_type == 'MCQ(multiple)':
            for let in extracted_ans:
                if let not in choices:
                    flag = 0
                    break
            extracted_ans = extracted_ans.upper()
        else:
            try:
                float_ans = float(extracted_ans)
            except Exception as e:
                flag = 0
        if flag == 1:
            return extracted_ans
        else:
            return 'None'
    else:
        answer = answer.strip().upper().replace(' ', '').replace(',', '').replace('AND', '').replace(':', '')
        print(f'Processed strings:{answer}\n')
        match1 = re.findall(r'[\[,\{,\(][A-D]+[\],\},\)]', answer)
        match2 = re.findall(r'[\[,\{,\(]-?[0-9]+\.?[0-9]*[\],\},\)]', answer)
        match3 = re.findall(r'ANSWERIS-?[0-9]+\.?[0-9]*', answer)
        match4 = re.findall(r'ANSWERIS[A-D]{1,4}', answer)
        match5 = re.findall(r'ANSWER-?[0-9]+\.?[0-9]*', answer)
        match6 = re.findall(r'ANSWER[A-D]{1,4}', answer)
        match7 = re.findall(
            r'[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]',
            answer)
        match8 = re.findall(r'[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]', answer)
        match9 = re.findall(r'[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]', answer)
        match10 = re.findall(r'ANSWERIS[\[,\{,\(]-?[0-9]+\.?[0-9]*[\],\},\)]', answer)
        match11 = re.findall(r'ANSWER[\[,\{,\(]-?[0-9]+\.?[0-9]*[\],\},\)]', answer)
        match12 = re.findall(r'ANSWERIS[\[,\{,\(][A-D]{1,4}[\],\},\)]', answer)
        match13 = re.findall(r'ANSWER[\[,\{,\(][A-D]{1,4}[\],\},\)]', answer)
        match14 = re.findall(
            r'ANSWERIS[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]',
            answer)
        match15 = re.findall(r'ANSWERIS[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]',
                             answer)
        match16 = re.findall(r'ANSWERIS[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]', answer)
        match17 = re.findall(
            r'ANSWER[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]',
            answer)
        match18 = re.findall(r'ANSWER[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]',
                             answer)
        match19 = re.findall(r'ANSWER[\[,\{,\(][A-D][\],\},\)][\[,\{,\(][A-D][\],\},\)]', answer)

        if match14:
            print('Answer matching type 14\n')
            ans = match14[-1]
            ans = ans[8:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match15:
            print('Answer matching type 15\n')
            ans = match15[-1]
            ans = ans[8:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match16:
            print('Answer matching type 16\n')
            ans = match16[-1]
            ans = ans[8:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match12:
            print('Answer matching type 12\n')
            ans = match12[-1]
            ans = ans[8:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match17:
            print('Answer matching type 17\n')
            ans = match17[-1]
            ans = ans[6:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match18:
            print('Answer matching type 18\n')
            ans = match18[-1]
            ans = ans[6:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match19:
            print('Answer matching type 19\n')
            ans = match19[-1]
            ans = ans[6:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match13:
            print('Answer matching type 13\n')
            ans = match13[-1]
            ans = ans[6:]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match10:
            print('Answer matching type 10\n')
            ans = match10[-1]
            ans = ans[9:]
            ans = ans[:-1]
            if 'MCQ' not in q_type:
                try:
                    float_ans = float(ans)
                    return ans
                except Exception as e:
                    print('Matching error!\n')

        if match11:
            print('Answer matching type 11\n')
            ans = match11[-1]
            ans = ans[7:]
            ans = ans[:-1]
            if 'MCQ' not in q_type:
                try:
                    float_ans = float(ans)
                    return ans
                except Exception as e:
                    print('Matching error!\n')

        if match3:
            print('Answer matching type 3\n')
            ans = match3[-1]
            ans = ans[8:]
            if 'MCQ' not in q_type:
                try:
                    float_ans = float(ans)
                    return ans
                except Exception as e:
                    print('Matching error!\n')

        if match4:
            print('Answer matching type 4\n')
            ans = match4[-1]
            ans = ans[8:]
            if 'MCQ' in q_type:
                return ans

        if match5:
            print('Answer matching type 5\n')
            ans = match5[-1]
            ans = ans[6:]
            if 'MCQ' not in q_type:
                try:
                    float_ans = float(ans)
                    return ans
                except Exception as e:
                    print('Matching error!\n')

        if match6:
            print('Answer matching type 6\n')
            ans = match6[-1]
            ans = ans[6:]
            if 'MCQ' in q_type:
                return ans

        if match7:
            print('Answer matching type 7\n')
            ans = match7[-1]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match8:
            print('Answer matching type 8\n')
            ans = match8[-1]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match9:
            print('Answer matching type 9\n')
            ans = match9[-1]
            final_ans = ''
            for let in ans:
                if let in choices:
                    final_ans = final_ans + let
            if 'MCQ' in q_type:
                return final_ans

        if match1:
            print('Answer matching type 1\n')
            ans = match1[-1]
            ans = ans[1:]
            ans = ans[:-1]
            if 'MCQ' in q_type:
                return ans

        if match2:
            print('Answer matching type 2\n')
            ans = match2[-1]
            ans = ans[1:]
            ans = ans[:-1]
            if 'MCQ' not in q_type:
                try:
                    float_ans = float(ans)
                    return ans
                except Exception as e:
                    print('Matching error!\n')
        print('answer invalid!\n')
        return 'None'
