from transformers import AutoTokenizer, BertModel, BertTokenizer
import torch

tokenizer = BertTokenizer.from_pretrained("bert-base-cased")
model = BertModel.from_pretrained("bert-base-uncased")

sequence_a = "This is a short sequence."
sequence_b = "This is a rather long sequence. It is at least longer than the sequence A."

encoded_sequence_a = tokenizer(sequence_a)["input_ids"]
encoded_sequence_b = tokenizer(sequence_b)["input_ids"]
padded_sequences = tokenizer([sequence_a, sequence_b], padding=True)
tokenizer.decode(padded_sequences["input_ids"][0])
tokenizer.decode(padded_sequences["input_ids"][1])

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
outputs = model(**inputs)

last_hidden_states = outputs.last_hidden_state