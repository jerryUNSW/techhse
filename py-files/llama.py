# export CUDA_VISIBLE_DEVICES=2,3

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_checkpoint = "/data1/llm_61/llama3_8b"

# Use device_map='auto' for multi-GPU model parallelism
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Running on {device}")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

# Load model with model parallelism across multiple GPUs
model = AutoModelForCausalLM.from_pretrained(
    model_checkpoint,
    torch_dtype=torch.float16,  # Use FP16 to reduce memory usage
    device_map="auto"           # Automatically distribute model across GPUs
)

# (Optional) Generate a sample output
prompt = "Explain Laplace mechanism in simple terms. Be concise"
inputs = tokenizer(prompt, return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_new_tokens=150)

# Print the result
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
