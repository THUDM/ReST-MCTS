import os
import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM


# get model and tokenizer
def get_inference_model(model_dir):
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    inference_model = AutoModel.from_pretrained(model_dir, trust_remote_code=True).half().cuda()
    inference_model.eval()
    return inference_tokenizer, inference_model


# get llama model and tokenizer
def get_inference_model_llama(model_dir):
    inference_model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    device = "cuda"
    inference_model.to(device)
    return inference_tokenizer, inference_model


# get mistral model and tokenizer
def get_inference_model_mistral(model_dir):
    inference_model = AutoModelForCausalLM.from_pretrained(model_dir, trust_remote_code=True, torch_dtype=torch.bfloat16)
    inference_tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    # inference_tokenizer.pad_token = inference_tokenizer.eos_token
    device = "cuda"
    inference_model.to(device)
    return inference_tokenizer, inference_model


# get glm model response
def get_local_response(query, model, tokenizer, max_length=2048, truncation=True, do_sample=False, max_new_tokens=1024, temperature=0.7):
    cnt = 2
    all_response = ''
    while cnt:
        try:
            inputs = tokenizer([query], return_tensors="pt", truncation=truncation, max_length=max_length).to('cuda')
            output_ = model.generate(**inputs, do_sample=do_sample, max_new_tokens=max_new_tokens, temperature=temperature)
            output = output_.tolist()[0][len(inputs["input_ids"][0]):]
            response = tokenizer.decode(output)

            print(f'obtain response:{response}\n')
            all_response = response
            break
        except Exception as e:
            print(f'Error:{e}, obtain response again...\n')
            cnt -= 1
    if not cnt:
        return []
    split_response = all_response.strip().split('\n')
    return split_response


# get llama model response
def get_local_response_llama(query, model, tokenizer, max_length=2048, truncation=True, max_new_tokens=1024, temperature=0.7, do_sample=False):
    cnt = 2
    all_response = ''
    # messages = [{"role": "user", "content": query}]
    # data = tokenizer.apply_chat_template(messages, return_tensors="pt").cuda()
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]
    message = '<|start_header_id|>user<|end_header_id|>\n\n{query}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n'.format(query=query)
    data = tokenizer.encode_plus(message, max_length=max_length, truncation=truncation, return_tensors='pt')
    input_ids = data['input_ids'].to('cuda')
    attention_mask = data['attention_mask'].to('cuda')
    while cnt:
        try:
            # query = "<s>Human: " + query + "</s><s>Assistant: "
            # input_ids = tokenizer([query], return_tensors="pt", add_special_tokens=False).input_ids.to('cuda')
            output = model.generate(input_ids, attention_mask=attention_mask, do_sample=do_sample, max_new_tokens=max_new_tokens, temperature=temperature, eos_token_id=terminators, pad_token_id=tokenizer.eos_token_id)
            ori_string = tokenizer.decode(output[0], skip_special_tokens=False)
            processed_string = ori_string.split('<|end_header_id|>')[2].strip().split('<|eot_id|>')[0].strip()
            response = processed_string.split('<|end_of_text|>')[0].strip()

            # print(f'获得回复:{response}\n')
            all_response = response
            break
        except Exception as e:
            print(f'Error:{e}, obtain response again...\n')
            cnt -= 1
    if not cnt:
        return []
    # split_response = all_response.split("Assistant:")[-1].strip().split('\n')
    split_response = all_response.split('\n')
    return split_response


# get mistral model response
def get_local_response_mistral(query, model, tokenizer, max_length=1024, truncation=True, max_new_tokens=1024, temperature=0.7, do_sample=False):
    cnt = 2
    all_response = ''
    # messages = [{"role": "user", "content": query}]
    # data = tokenizer.apply_chat_template(messages, max_length=max_length, truncation=truncation, return_tensors="pt").cuda()
    message = '[INST]' + query + '[/INST]'
    data = tokenizer.encode_plus(message, max_length=max_length, truncation=truncation, return_tensors='pt')
    input_ids = data['input_ids'].to('cuda')
    attention_mask = data['attention_mask'].to('cuda')
    while cnt:
        try:
            output = model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, do_sample=do_sample, temperature=temperature, eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
            ori_string = tokenizer.decode(output[0])
            processed_string = ori_string.split('[/INST]')[1].strip()
            response = processed_string.split('</s>')[0].strip()

            print(f'obtain response:{response}\n')
            all_response = response
            break
        except Exception as e:
            print(f'Error:{e}, obtain response again...\n')
            cnt -= 1
    if not cnt:
        return []
    all_response = all_response.split('The answer is:')[0].strip()  # intermediate steps should not always include a final answer
    ans_count = all_response.split('####')
    if len(ans_count) >= 2:
        all_response = ans_count[0] + 'Therefore, the answer is:' + ans_count[1]
    all_response = all_response.replace('[SOL]', '').replace('[ANS]', '').replace('[/ANS]', '').replace('[INST]', '').replace('[/INST]', '').replace('[ANSW]', '').replace('[/ANSW]', '')  # remove unique answer mark for mistral
    split_response = all_response.split('\n')
    return split_response
