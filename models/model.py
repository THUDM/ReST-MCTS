import os
import openai
import backoff
import requests
import json
from models.inference_models import get_local_response, get_inference_model, get_inference_model_llama, get_local_response_llama, get_inference_model_mistral, get_local_response_mistral
from models.value_models import get_local_value, get_value_model, get_value_model_prm, get_value_model_mistral, get_value_model_prm_mistral
from transformers import AutoModel, AutoTokenizer

# openai api settings
API_KEY = 'sk-**'
API_BASE = 'base'
BASE_MODEL_GPT = "gpt-3.5-turbo"

# GLM api settings
URL = "https://api.chatglm.cn/v1/chat/completions"
ID = "**"
AUTH = '**'
CONTENT_TYPE = 'application/json; charset=utf-8'
BASE_MODEL_GLM = 'GLM4'

# local model settings
# if you want to use local models, set these two directories

INFERENCE_MODEL_DIR = None
LOCAL_INFERENCE_TYPES = ['glm', 'llama', 'mistral']
LOCAL_INFERENCE_IDX = 0

VALUE_BASE_MODEL_DIR = None
VALUE_MODEL_STATE_DICT = None
LOCAL_VALUE_TYPES = ['glm', 'mistral']
LOCAL_VALUE_IDX = 0
USE_PRM = False

INFERENCE_LOCAL = False
VALUE_LOCAL = False

# implement the inference model
if INFERENCE_MODEL_DIR is not None:
    INFERENCE_LOCAL = True
    inference_type = LOCAL_INFERENCE_TYPES[LOCAL_INFERENCE_IDX]
    if inference_type == 'glm':
        inference_tokenizer, inference_model = get_inference_model(INFERENCE_MODEL_DIR)
    elif inference_type == 'llama':
        inference_tokenizer, inference_model = get_inference_model_llama(INFERENCE_MODEL_DIR)
    else:
        inference_tokenizer, inference_model = get_inference_model_mistral(INFERENCE_MODEL_DIR)

# implement the value model(reward model)
if VALUE_BASE_MODEL_DIR is not None:
    VALUE_LOCAL = True
    value_type = LOCAL_VALUE_TYPES[LOCAL_VALUE_IDX]
    if USE_PRM:
        if value_type == 'glm':
            value_tokenizer, value_model = get_value_model_prm(VALUE_BASE_MODEL_DIR, VALUE_MODEL_STATE_DICT)
        else:
            value_tokenizer, value_model = get_value_model_prm_mistral(VALUE_BASE_MODEL_DIR, VALUE_MODEL_STATE_DICT)
    else:
        if value_type == 'glm':
            value_tokenizer, value_model = get_value_model(VALUE_BASE_MODEL_DIR, VALUE_MODEL_STATE_DICT)
        else:
            value_tokenizer, value_model = get_value_model_mistral(VALUE_BASE_MODEL_DIR, VALUE_MODEL_STATE_DICT)

completion_tokens = prompt_tokens = 0
api_key = API_KEY
if api_key != "":
    openai.api_key = api_key
    print(f'api_key:{api_key}\n')
else:
    print("Warning: OPENAI_API_KEY is not set")

api_base = API_BASE
if api_base != "":
    print("Warning: OPENAI_API_BASE is set to {}".format(api_base))
    openai.api_base = api_base


@backoff.on_exception(backoff.expo, openai.error.OpenAIError)
def completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


def gpt(prompt, model=BASE_MODEL_GPT, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    messages = [{"role": "user", "content": prompt}]
    out = []
    cnt = 5
    while cnt:
        try:
            out = chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)[
                0].split('\n')
            break
        except Exception as e:
            print(f"Error occurred when getting gpt reply!\nError type:{e}\n")
            cnt -= 1
    return out


def chatgpt(messages, model=BASE_MODEL_GPT, temperature=0.7, max_tokens=1000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
                                       n=cnt, stop=stop)
        # print(f'得到GPT回复:{res}\n\n')
        outputs.extend([choice["message"]["content"] for choice in res["choices"]])
        # log completion tokens
        completion_tokens += res["usage"]["completion_tokens"]
        prompt_tokens += res["usage"]["prompt_tokens"]
    return outputs


def gpt_usage(backend=BASE_MODEL_GPT):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    else:
        cost = -1
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}


def extract_data(text):
    lines = text.split('\n')
    extracted_data = []
    should_extract = False

    for line in lines:
        if line.startswith('event: finish'):
            should_extract = True
        elif should_extract and line.startswith('data: '):  # and "left: " in line:
            if len(line[6:]) > 0:
                extracted_data.append(line[6:])  # Remove 'data: ' prefix, remain '\n'
    return extracted_data


def glm(prompt, model=BASE_MODEL_GLM, temperature=0.7, max_tokens=1000, seed=170) -> list:
    return get_glm_reply(prompt, model, temperature=temperature, max_tokens=max_tokens, seed=seed)


def get_glm_reply(query, model, temperature=0.7, max_tokens=1000, seed=175):
    if model == 'ChatGLM2':
        url = URL
        payload = {
            "id": ID,
            "prompt": query,
            "seed": seed,
            "max_tokens": str(max_tokens),
            "temperature": temperature,
        }
        headers = {
            'Authorization': AUTH,
            'Content-Type': CONTENT_TYPE
        }

        tol = 3
        response = None
        while tol:
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                break
            except Exception as e:
                print(f'Error occurred when getting proxy reply!\nError type:{e}\nRetrying...\n')
                tol -= 1

        if response is None:
            print('Error occurred when getting proxy reply!\n')
            return []

        reply = response.content.decode('utf-8')
        replies = extract_data(reply)
        return replies

    elif model == 'GLM4':
        url = URL
        payload = {
            'model': "glm4-alltools-130b-awq",
            "messages": [{"role": "user", "content": query}],
            "temperature": temperature,
            "top_p": 0.7,
            "stream": False,
            "max_tokens": max_tokens
        }
        headers = {
            'Authorization': AUTH,
            'Content-Type': CONTENT_TYPE
        }

        tol = 3
        response = None
        while tol:
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                break
            except Exception as e:
                print(f'Error occurred when getting proxy reply!\nError type:{e}\nRetrying...\n')
                tol -= 1

        if response is None:
            print('Error occurred when getting proxy reply!\n')
            return []

        reply = response.content.decode('utf-8')
        # print('reply:', reply)
        try:
            content = reply.split("\"content\":\"")[1].split("\",\"role\":\"assistant\"")[0]
        except Exception as e:
            print(f'Error occurred when decoding reply!\nError type:{e}\n')
            return []
        return content.split('\n')

    elif model == 'GLM3':
        url = URL
        payload = {
            'model': "chatglm3-32b-v0.8",
            "messages": [{"role": "user", "content": query}],
            "temperature": temperature,
            "top_p": 0.7,
            "stream": False,
            "max_tokens": max_tokens
        }
        headers = {
            'Authorization': AUTH,
            'Content-Type': CONTENT_TYPE
        }

        tol = 3
        response = None
        while tol:
            try:
                response = requests.post(url, headers=headers, data=json.dumps(payload))
                break
            except Exception as e:
                print(f'Error occurred when getting proxy reply!\nError type:{e}\nRetrying...\n')
                tol -= 1

        if response is None:
            print('Error occurred when getting proxy reply!\n')
            return []

        reply = response.content.decode('utf-8')
        # print('reply:', reply)
        try:
            content = reply.split("\"content\":\"")[1].split("\",\"role\":\"assistant\"")[0]
        except Exception as e:
            print(f'Error occurred when decoding reply!\nError type:{e}\n')
            return []
        return content.split('\n')

    else:
        print('Unsupported glm model!\n')
        return []


def local_inference_model(query, max_length=2048, truncation=True, do_sample=False, max_new_tokens=1024,
                          temperature=0.7):
    assert INFERENCE_LOCAL, "Inference model not implemented!\n"
    if inference_type == 'glm':
        return get_local_response(query, inference_model, inference_tokenizer, max_length=max_length,
                                  truncation=truncation,
                                  do_sample=do_sample, max_new_tokens=max_new_tokens, temperature=temperature)
    elif inference_type == 'llama':
        return get_local_response_llama(query, inference_model, inference_tokenizer, max_new_tokens=max_new_tokens,
                                        temperature=temperature, do_sample=do_sample)
    else:
        return get_local_response_mistral(query, inference_model, inference_tokenizer, max_new_tokens=max_new_tokens,
                                          temperature=temperature, do_sample=do_sample)


def local_value_model(prompt_answer, max_length=2048, low=0, high=1):
    assert VALUE_LOCAL, "Value model not implemented!\n"
    return get_local_value(prompt_answer, value_model, value_tokenizer, max_length=max_length, low=low, high=high)
