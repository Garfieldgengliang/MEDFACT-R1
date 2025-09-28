from transformers import AutoProcessor
from vllm import LLM, SamplingParams
from qwen_vl_utils import process_vision_info
import json
# MODEL_PATH = "/root/autodl-tmp/QwenVL-3B-IUxray-fakelabel-consistentreward"
MODEL_PATH = "/root/autodl-tmp/QwenVL-3B-IUxray-sft-grpo"
import logging
# default: Load the model on the available device(s)
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "/root/autodl-tmp/Qwen2-5-VL-3B-Instruct", torch_dtype="auto", device_map="cuda:0"
# )
logging.basicConfig(filename='/root/autodl-tmp/R1-V-inference/iuxray_test_inference_0527.log', level=logging.INFO)

logging.log(logging.INFO, 'Start of inferencing IUxray test dataset')

llm = LLM(
    model=MODEL_PATH,
    # limit_mm_per_prompt={"image": 10, "video": 10},
)

sampling_params = SamplingParams(
    temperature=0.9,
    top_p=0.5,
    repetition_penalty=1.05,
    max_tokens=2048,
    stop_token_ids=[],
)
min_pixels = 256*28*28
max_pixels = 501760
processor = AutoProcessor.from_pretrained("/root/autodl-tmp/QwenVL-3B-IUxray-sft-grpo", min_pixels=min_pixels, max_pixels=max_pixels)

with open('grpo_iuxray_test_data_20250411.json', 'r') as file:
    grpo_test_dict = json.loads(file.read())
image_path_list = grpo_test_dict['image_path']
question_list = grpo_test_dict['question']

inference_answer_list = []

for index in range(len(question_list)):
    print('current inferencing index is ', index)
    ques = question_list[index]
    img = image_path_list[index]

    image_messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": img,
                    "min_pixels": min_pixels,
                    "max_pixels": max_pixels,
                    # "image": "/root/autodl-tmp/mimic_used/p18/p18716770/s54683735/6d1be0e0-f5bb49be-a11fef79-97b98c05-b3089091.jpg",
                },
                {"type": "text", "text": f"{ques} A conversation between User and Medical Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>"},
            ],
        }
    ]


    # Here we use video messages as a demonstration
    # messages = image_messages

    processor = AutoProcessor.from_pretrained(MODEL_PATH)
    prompt = processor.apply_chat_template(
        image_messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    image_inputs, video_inputs, video_kwargs = process_vision_info(image_messages, return_video_kwargs=True)

    mm_data = {}
    if image_inputs is not None:
        mm_data["image"] = image_inputs
    if video_inputs is not None:
        mm_data["video"] = video_inputs

    llm_inputs = {
        "prompt": prompt,
        "multi_modal_data": mm_data,

        # FPS will be returned in video_kwargs
        "mm_processor_kwargs": video_kwargs,
    }

    outputs = llm.generate([llm_inputs], sampling_params=sampling_params)
    generated_text = outputs[0].outputs[0].text

    print(generated_text)

    # inference_answer_list.append(generated_text)
    logging.log(logging.INFO, generated_text)
    inference_answer_list.append(generated_text)

assert len(inference_answer_list) == len(question_list)

iuxray_test_grpo_data_answer = {
    'question': question_list,
    'label': grpo_test_dict['answer'],
    'answer': inference_answer_list
}
with open("/root/autodl-tmp/R1-V-inference/grpo_iuxray_test_answer_20250527.json", "w", encoding="utf-8") as f:
    json.dump(iuxray_test_grpo_data_answer, f, ensure_ascii=False, indent=4)

    