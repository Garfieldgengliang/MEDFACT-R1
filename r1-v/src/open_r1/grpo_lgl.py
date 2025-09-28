# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from openai import OpenAI

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format", "fact", "consistent"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'fact', 'consistent'"},
        # metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format', 'fact"},
    )

    # reward_funcs: list[str] = field(
    #     default_factory=lambda: ["accuracy", "format", ],
    #     metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format',"},
    # )


    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def accuracy_reward(completions, solution, **kwargs):
    """Reward function that checks if the completion is correct using either symbolic verification or exact string matching."""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(contents, solution):
        reward = 0.0
        # Try symbolic verification first
        try:
            # answer = parse(content)
            # if float(verify(answer, parse(sol))) > 0:
            #     reward = 1.0
            if sol.lower() in content.lower():
                reward = 1.0
        except Exception:
            pass  # Continue to next verification method if this fails

        # If symbolic verification failed, try string matching
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol)
                ground_truth = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags
                content_match = re.search(r'<answer>(.*?)</answer>', content)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Compare the extracted answers
                if student_answer == ground_truth:
                    reward = 1.0
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
                
        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Solution: {sol}\n")
    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
    completion_contents = [completion[0]["content"] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]

def fact_reward(completions, keyword, **kwargs):
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, keys in zip(contents, keyword):
        reward = 0.0
        try:
            key_list = keys.split("/")
            for k_word in key_list:
                if k_word.lower() in content.lower():
                    reward += 0.2
        except Exception:
            pass

        rewards.append(reward)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH")
            # local_rank = int(os.getenv("LOCAL_RANK", 0))
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(f"------------- {current_time} Fact reward: {reward} -------------\n")
                f.write(f"Content: {content}\n")
                f.write(f"Keyword: {keys}\n")
    print('current fact rewards list is ', rewards)
    return rewards

def logical_consistency_check(think, answer):
    """Check if the logical part of the think process is consistent with the answer."""
    # Implement your logic consistency check here
    # For example, you can use a simple keyword match or a more complex NLP model
    # For now, let's just return True if the think part is in the answer
    API_SECRET_KEY = 'sk-5ac0871bcbb3476e82a17b6bb0e0691b'
    BASE_URL = 'https://dashscope.aliyuncs.com/compatible-mode/v1'
    # # MODEL_NAME = 'qwen2.5-7b-instruct-ft-202504020954-b784'
    MODEL_NAME = 'qwen2.5-72b-instruct'
    prompt_qua = "I will provide you with a reasoning process and a corresponding answer. You need to tell me if the reasoning process can lead to the answer. For example, \
        if the reasoning process said '<think> Pneumothorax is a condition where air accumulates in the pleural space, which is the area between the lungs and the chest wall. \
        On an X-ray, this can appear as a dark or white area in the lung field, depending on the location of the air collection. However, the image provided does not show \
        any obvious signs of pneumothorax. It appears to be a normal chest X-ray without any visible abnormality that would suggest pneumothorax. Therefore, \
        based on the given image, pneumothorax cannot be definitively seen.</think>' and the answer said '<answer> yes, pneumothorax can be identified.</answer>', it means \
          the reasoning process does not match with the answer, thus you should respond 'No'. On the other hand, you should respond 'Yes'. Now here is the reasoning process{} \n here is the answer {} \n your response: ".format(think, answer)
    gpt4_client = OpenAI(api_key=API_SECRET_KEY, base_url=BASE_URL)
    completion = gpt4_client.chat.completions.create(
        model = MODEL_NAME,
        messages=[
            {'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': prompt_qua}],
        top_p=0.8,
        max_tokens=1024,
        temperature=0.2,
        # stream=True,
    )
    response = completion.to_dict()['choices'][0]['message']['content']
    print(response)
    return response

def consistent_reward(completions, **kwargs):
    """Reward function that checks if the logical part of think process is consistent with the answer."""
    contents = [completion[0]["content"] for completion in completions]
    pattern_1 = r"<think>(.*?)</think>"
    pattern_2 = r"<answer>(.*?)</answer>"
    rewards = []
    # current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content in contents:
        reward = 0.0
        # try:
        result_1 = re.search(pattern_1, content)
        result_2 = re.search(pattern_2, content)
        if result_1 and result_2 :
            print('matching think and answer pattern')
            think = '<think>' + result_1.group(1).strip() + '</think>'
            answer = '<answer>' + result_2.group(1).strip() + '</answer>'
            # answer = result_2.group(1).strip()
            print("think: ", think)
            print("answer: ", answer)
            # Check if the logical part of think process is consistent with the answer
            checker_response = logical_consistency_check(think, answer)
            if 'yes' in checker_response.lower()[:10]:
                reward = 1.2
            elif 'no' in checker_response.lower()[:10]:
                reward = -0.2
            else:
                reward = 0.2
            print('current reward is ', reward)
        else:
            print('No matching think and answer pattern')
            print('current reward is ', reward)
            # continue
        
        # except Exception:
            # pass

        rewards.append(reward)
        # 
        # if os.getenv("DEBUG_MODE") == "true":
        #     log_path = os.getenv("LOG_PATH")
        #     # local_rank = int(os.getenv("LOCAL_RANK", 0))
        #     with open(log_path, "a", encoding="utf-8") as f:
        #         f.write(f"------------- {current_time} Consistent reward: {reward} -------------\n")
                # f.write(f"Content: {content}\n")
    print('current consistent rewards list is ', rewards)
    return rewards




reward_funcs_registry = {
    "accuracy": accuracy_reward,
    "format": format_reward,
    "fact": fact_reward,
    "consistent": consistent_reward,
}


# reward_funcs_registry = {
#     "accuracy": accuracy_reward,
#     "format": format_reward,
# }

SYSTEM_PROMPT = (
    "A conversation between User and Medical Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # Load the dataset
    # dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config, )
    dataset = load_from_disk(script_args.dataset_name)

    # Format into conversation
    def make_conversation(example):
        return {
            "prompt": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": example["question"]},
            ],
        }

    # def make_conversation_image(example):
    #     return {
    #         "prompt": [
    #             {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
    #             {
    #                 "role": "user",
    #                 "content": [
    #                     {"type": "image"},
    #                     {"type": "text", "text": example["problem"]},
    #                 ],
    #             },
    #         ],
    #     }

    # QUESTION_TEMPLATE = "{Question}  Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
    QUESTION_TEMPLATE = "{Question} A conversation between User and Medical Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"
    def make_conversation_image(example):
        return {
            "prompt": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(Question=example["question"])},
                    ],
                },
            ],
        }

    
    print('SCRITP ARGS ', script_args.dataset_train_split)
    print('dataset features', dataset[script_args.dataset_train_split].features)
    print('script_args.dataset_config', script_args.dataset_config)
    print('script_args.dataset_name',script_args.dataset_name)
    if "image" in dataset[script_args.dataset_train_split].features:
        print("ATTENTION! HAS IMAGE IN DATASET")
        dataset = dataset.map(make_conversation_image)  # Utilize multiprocessing for faster mapping
        # dataset = dataset.remove_columns(["original_question", "original_answer"])

    else:
        print("no image in dataset")
        dataset = dataset.map(make_conversation)
        # dataset = dataset.remove_columns("messages")

    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split] if training_args.eval_strategy != "no" else None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )

    # Train and push the model to the Hub
    trainer.train()

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
