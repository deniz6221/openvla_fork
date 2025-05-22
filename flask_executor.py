from flask import Flask, request, jsonify
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor
import torch
from peft import PeftModel


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

app = Flask(__name__)

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files or 'prompt' not in request.form:
        return jsonify({"error": "Missing image or prompt"}), 400
    sent_image = request.files['image']
    prompt = request.form['prompt']
    image = Image.open(sent_image.stream).convert("RGB")
    
    inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)
    with torch.no_grad():
        action = lora_model.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)

    # Return result
    return jsonify({"action": action.tolist()})




if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
