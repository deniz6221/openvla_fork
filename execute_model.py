from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image
import torch
from peft import PeftModel

# Load Processor & VLA
processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
vla = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    attn_implementation="flash_attention_2",  # [Optional] Requires flash_attn
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    trust_remote_code=True
).to("cuda:0")

lora_model = PeftModel.from_pretrained(vla, "adapter_dir/openvla-7b+xarm_vla_dataset+b4+lr-0.0005+lora-r32+dropout-0.0--image_aug").to("cuda:0")

lora_model.merge_and_unload()


# Grab image input & format prompt
image: Image.Image = Image.open("test_data/test_1.png")

instruction = "put the black cup on top of the white plate"
prompt = f"In: What action should the robot take to {instruction}?\nOut:"

# Predict Action (7-DoF; un-normalize for BridgeData V2)
inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
action = lora_model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

print(f"Action: {action}")