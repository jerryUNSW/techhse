from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import torch

model_name = "Qwen/Qwen2.5-7B-Instruct-1M"

# os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained(model_name)

prompt = """
Generate 10 replacements for the phrase 'LAS VEGAS' in the given text. Only show the replaced phrases as a list, one per line. 

Example:
Text: 'LAS VEGAS is known for its vibrant nightlife.'
Replace Phrases:
Reno
Atlantic City
Miami

Task:
Text: 'LAS VEGAS, Nevada (CNN) -- Former football star O.J. Simpson will be held without bail after his arrest on robbery and assault charges, police announced late Sunday. Police released this mug shot of O.J. Simpson'
Replace Phrases:
"""

messages = [
    {"role": "system", "content": "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

print("response = ", response)