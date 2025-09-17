from datasets import Dataset, Features, Image, Value

import json

with open('grpo_data_20250303.json', 'r') as file:
    grpo_dict = json.loads(file.read())
image_path_list = grpo_dict['image_path']
new_image_path_list = []
for item in image_path_list:
    new_image_path_list.append('/root/autodl-tmp/mimic_used/'+ item)
    


features = Features({
    "question_id": Value("int32"),
    "image": Image(),
    "question": Value("string"),
    "answer": Value("string"),
    "semantic_type":Value("string"),
    "content_type":Value("string")
    
})

dataset = Dataset.from_dict({
    "question_id": list(range(len(new_image_path_list))),
    "image": new_image_path_list,
    "question": grpo_dict['question'],
    "answer": grpo_dict['answer'],
    "semantic_type":grpo_dict['semantic_type'],
    "content_type":grpo_dict['content_type']
}, features=features)

split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
split_dataset.save_to_disk("/root/autodl-tmp/mimic_grpo_0306")
# train_data = split_dataset["train"]
# test_data = split_dataset["test"]

# train_data.save_to_disk("/root/autodl-tmp/mimic_grpo_0304/train")
# test_data.save_to_disk("/root/autodl-tmp/mimic_grpo_0304/test")
