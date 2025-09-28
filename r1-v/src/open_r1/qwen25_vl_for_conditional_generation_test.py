from transformers import Qwen2_5_VLForConditionalGeneration
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    '/root/autodl-tmp/Qwen2-5-VL-3B-Instruct', local_files_only=True, 
                )

