import os
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", return_dict=False)
model.eval()
save_folder = "bert-classification"
os.makedirs(save_folder, exist_ok=True)

dummpy_text = "안녕하세요."

tokenized_text = tokenizer(dummpy_text, return_tensors='pt', max_length=256, padding="max_length")

traced = torch.jit.trace(model, (tokenized_text['input_ids'], tokenized_text['attention_mask']))
torch.jit.save(traced, os.path.join(save_folder, "model.pt"))
tokenizer.save_pretrained(save_folder)
