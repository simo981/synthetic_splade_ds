import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import json
import dill

model_name = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

documents = []
with open("synt_docs.pkl", "rb") as file:
    documents = dill.load(file)
queries = []
with open("synt_queries.pkl", "rb") as file:
    queries = dill.load(file)
  
def splade_encode(text):
    with torch.no_grad():
        inputs = tokenizer(
        text,
        max_length=512,
        truncation=True,
        padding='max_length',
        return_tensors='pt')        
        outputs = model(**inputs).logits[0]
        sparse_vec = torch.max(outputs, dim=0)[0]
        sparse_dict = {}
        for idx, weight in enumerate(sparse_vec):
            val = weight.item()
            if val > 0.01:  # Thresh on .x
                token_name = tokenizer.convert_ids_to_tokens(idx)
                sparse_dict[token_name] = float(val)
        return sparse_dict

docs_splade = []
for doc_id, doc_text in enumerate(documents):
    vec = splade_encode(doc_text)
    docs_splade.append({
        "id": doc_id,
        "content": doc_text,
        "vector": vec
    })

query_splade = []
for query_id, query_text in enumerate(queries):
    vec = splade_encode(query_text)
    query_splade.append({
        "id": query_id,
        "content": query_text,
        "vector": vec
    })

with open("my_splade_docs.jsonl", "w", encoding="utf-8") as f:
    for obj in docs_splade:
        f.write(json.dumps(obj) + "\n")

with open("my_splade_queries.jsonl", "w", encoding="utf-8") as f:
    for obj in query_splade:
        f.write(json.dumps(obj) + "\n")
