from transformers import AutoModelForVision2Seq, AutoProcessor
from PIL import Image

import torch

model = AutoModelForVision2Seq.from_pretrained(
    "openvla/openvla-7b",
    cache_dir="./vla_dir",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    low_cpu_mem_usage=True,
    attn_implementation="flash_attention_2",
).to("cuda:0")

processor = AutoProcessor.from_pretrained(
    "openvla/openvla-7b", 
    cache_dir="./vla_dir", 
    trust_remote_code=True
)