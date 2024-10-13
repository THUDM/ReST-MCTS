# ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search

<p align="center">
📃 <a href="https://arxiv.org/abs/2406.03816" target="_blank">[ReST-MCTS*]</a> 
<a href="https://github.com/THUDM/ReST-MCTS" target="_blank">[GitHub]</a>
<a href="https://rest-mcts.github.io/" target="_blank">[Website]</a> <br>
</p>

We develop a reinforced self-training approach, called **ReST-MCTS***, based on integrating process reward guidance with tree search MCTS* for collecting higher-quality reasoning traces as well as per-step value to train policy and reward models. **ReST-MCTS*** circumvents the per-step manual annotation typically used to train process rewards by tree-search-based reinforcement learning: Given oracle final correct answers, **ReST-MCTS*** is able to infer the correct process rewards by estimating the probability this step can help lead to the correct answer. These inferred rewards serve dual purposes: they act as value targets for further refining the process reward model and also facilitate the selection of high-quality traces for policy model self-training.

![](./assets/overall.png)

## **Table of Contents**

- [Key Differences](#introduction)
- [Getting Started](#started)
- [Data & Model](#data&model)
- [Self-training](#Self-training)
- [Leaderboard](#Leaderboard)
- [Citation](#Citation)

## **Key Differences**
![](./assets/comparison.png)

## **Getting Started**

### **Prepare Env**
You should install the required packages by running
```bash
pip install -r requirements.txt
```
Note that for some models on huggingface like the GLM series, you may need to install specific versions of `transformers`.

### **Model Implementation**
To run MCTS* search, you should implement a policy as well as a process reward model (value model).
You can directly set these models by providing the model paths in the file `models/model.py`, substituting `INFERENCE_MODEL_DIR`, `VALUE_BASE_MODEL_DIR` and `VALUE_MODEL_STATE_DICT`.

`INFERENCE_MODEL_DIR` is the local path to the policy model, model could be Llama3-8b-Instruct, Mistral-7B: MetaMATH, 
and SciGLM-6B.
`VALUE_BASE_MODEL_DIR` is the local path to the value model. Considering the different dependency versions of `transformers`, Mistral-7B is adopted as the backbone of the value model when the policy model is Llama3-8B-Instruct or Mistral-7B: MetaMATH. When the policy model is SciGLM, we use ChatGLM3-6b-base as the backbone of the value model.
You can load the model and get the `VALUE_MODEL_STATE_DICT`.

We now only provide the implementation of the `llama`, `glm` and `mistral` as policy, with `glm` and `mistral` as value model.
If you are trying with other models, you can refer to our implementation and modify relevant codes to implement the corresponding models.
Once you've implemented the policy and value model, you should modify the `LOCAL_INFERENCE_IDX` and `LOCAL_VALUE_IDX` in `models/model.py` to the corresponding model index.

### **Data Preparation**
Before running search for evaluation or generation, you have to make sure your target question dataset is in the correct format. 
The data file should be a json file with items in the following format:
```json
{
  "content": "Calculate the sum of the first 10 prime numbers.",
  "answer": "129"
}
```
The `content` entry is required, serving as the question. While the `answer` entry is optional, it is used for evaluation.

### **Run MCTS\* Search**
The implementation of MCTS* search can be found in `MCTS`. We provide a search interface in `MCTS/task.py`. To run MCTS* search for a single question, you can refer to the following script:

```python
from MCTS.task import *
question = "Calculate the sum of the first 10 prime numbers."
task = MCTS_Task(question, 'llama', 'local', lang='en')
output = task.run()
print(output['solution'])
```

For evaluation of MCTS* on benchmarks, you can refer to `evalaute.py`, setting the parameter `--mode` to "mcts". You should specify the benchmark name and the exact file (subset) you want to evaluate. A simple demonstration is provided below:
```bash
python evaluate.py \
  --task_name "scibench" \
  --file "thermo" \
  --propose_method "gpt" \
  --value_method "local" \
  --mode "mcts" \
  --evaluate "scibench" \
  --iteration_limit 50 \
  --use_reflection "simple" \
  --branch 3
```
You can also refer to the `MCTS/args.md` for more details on the search parameters.

## **Data & Model (take Llama3-8B-Instruct as an example)**
Aiming to gather value train data for science, we integrate questions of a lean science dataset $D_{sci}$ within <a href="https://rest-mcts.github.io/" target="_blank">[SciInstruct]</a> to construct $D_{V_0}$. This dataset consists of 11,554 questions, where each question is paired with a correct step-by-step solution. (See Section 4.1 of the paper for more details.)
Then, we use $D_V0$ to train Mistral-7B: MetaMATH as the initial process reward model.

Given question set $D_G$, we use Llama3-8B-Instruct to generate synthetic data for policy model and value model. (See Algorithm 1 of the paper for more details.)

Download policy data (positive samples) for training 1st policy model (Llama3-8b-Instruct):
[[Hugging Face](https://huggingface.co/datasets/zd21/ReST-MCTS-Llama3-8b-Instruct-Policy-1st)]

Download PRM data (positive and negative samples) for training 1st reward model (Mistral-7B: MetaMATH):
[[Hugging Face](https://huggingface.co/datasets/zd21/ReST-MCTS-Llama3-8b-Instruct-PRM-1st)]

Download the trained policy model:
[[Hugging Face](https://huggingface.co/zd21/ReST-MCTS-Llama3-8b-Instruct-Policy-1st)]

## **Self-training**
For our methods:

Regarding Llama3-8B-Instruct and Mistral-7B: MetaMATH, we use the default repo of <a href="https://github.com/TIGER-AI-Lab/MAmmoTHhttps://github.com/TIGER-AI-Lab/MAmmoTH" target="_blank">[MAmmoTH]</a> to train the policy model and evaluate.

Regarding SciGLM-6B, we use the default repo of <a href="https://github.com/THUDM/SciGLM" target="_blank">[SciGLM]</a> to train the policy model and evaluate.

We also implement self-rewarding as our baseline in ./self_train/self_train_dpo.py.

## **Leaderboard**

Self-training Results:

![](./assets/results.png)

Accuracy of Different Verifiers:

![](./assets/vm_results.png)

Accuracy of Different Searches:

![](./assets/searches.png)

## **Citation**

If you find our work helpful, please kindly cite our paper:

```
@misc{zhang2024restmcts,
      title={ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search}, 
      author={Dan Zhang and Sining Zhoubian and Yisong Yue and Yuxiao Dong and Jie Tang},
      year={2024},
      eprint={2406.03816},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
