from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Caminho do modelo LLaMA 2
model_path = "./models/LLaMA-2-7b"

# Verificando se a GPU está disponível
device = "cuda" if torch.cuda.is_available() else "cpu"

# Carregando Tokenizer e Modelo
tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto", torch_dtype=torch.float16).to(device)

# Entrada de Exemplo
input_text = "Olá, como você está?"
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# Geração da Resposta
outputs = model.generate(**inputs, max_length=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
