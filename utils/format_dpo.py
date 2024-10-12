import json

round_n = 2
base = 'mistral'

dpo_dataset_dict = dict()
dpo_dataset_dict["prompt"] = list()
dpo_dataset_dict["chosen"] = list()
dpo_dataset_dict["rejected"] = list()

cnt = 0
with open(f'extracted_samples/self_train_{round_n}/cot/dpo-{base}_local.json',
          'rt') as f:
    for line in f.readlines():
        cont = json.loads(line)
        cnt += 1
        # print(cont, type(cont))
        dpo_dataset_dict["prompt"].append(cont['prompt'])
        dpo_dataset_dict["chosen"].append(cont['response_chosen'])
        dpo_dataset_dict["rejected"].append(cont['response_rejected'])

print(cnt)
print(len(dpo_dataset_dict["prompt"]))
print(len(dpo_dataset_dict["chosen"]))
print(len(dpo_dataset_dict["rejected"]))
out_file = f"extracted_samples/self_train_{round_n}/cot/{base}_local_dpo.json"

with open(out_file, "w") as f:
    json.dump(dpo_dataset_dict, f)
    print("Loading the file is complete...")

with open(out_file, 'r') as load_f:
    load_dict = json.load(load_f)
    print(len(load_dict['prompt']))
    print(len(load_dict['chosen']))
    print(len(load_dict['rejected']))
