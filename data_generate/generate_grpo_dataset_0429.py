from datasets import Dataset, Features, Image, Value

import json

with open('grpo_data_20250428_fake_label_harvard.json', 'r') as file:
    grpo_dict = json.loads(file.read())
image_path_list = grpo_dict['image_path']
# new_image_path_list = []
# for item in image_path_list:
#     new_image_path_list.append('/root/autodl-tmp/IUxray/images/'+ item)
# print(new_image_path_list)


features = Features({
    "question_id": Value("int32"),
    "image": Image(),
    "question": Value("string"),
    "solution": Value("string"),
    "keyword": Value("string"),
    
})

dataset = Dataset.from_dict({
    "question_id": list(range(len(image_path_list))),
    "image": image_path_list,
    "question": grpo_dict['question'],
    "solution": grpo_dict['solution'],
    "keyword": grpo_dict['keyword'],
}, features=features)

split_dataset = dataset.train_test_split(test_size=0.1, seed=42)
split_dataset.save_to_disk("/root/autodl-tmp/harvard_grpo_0429")
# train_data = split_dataset["train"]
# test_data = split_dataset["test"]

# train_data.save_to_disk("/root/autodl-tmp/mimic_grpo_0304/train")
# test_data.save_to_disk("/root/autodl-tmp/mimic_grpo_0304/test")
