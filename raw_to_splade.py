import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
import json
import dill
import concurrent.futures

model_name = "naver/splade-cocondenser-ensembledistil"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)
model.eval()

ths = 4

with open("synt_docs.pkl", "rb") as file:
    documents = dill.load(file)
with open("synt_queries.pkl", "rb") as file:
    queries = dill.load(file)

def splade_encode(text):
    """Encodes a piece of text using the SPLADE model and returns a sparse vector dictionary."""
    with torch.no_grad():
        inputs = tokenizer(
            text,
            max_length=512,
            truncation=True,
            padding='max_length',
            return_tensors='pt'
        )
        outputs = model(**inputs).logits[0]
        sparse_vec = torch.max(outputs, dim=0)[0]
        sparse_dict = {}
        for idx, weight in enumerate(sparse_vec):
            val = weight.item()
            if val > 0.01:  # threshold
                token_name = tokenizer.convert_ids_to_tokens(idx)
                sparse_dict[token_name] = float(val)
        return sparse_dict

def process_document(doc_tuple):
    doc_id, doc_text = doc_tuple
    vec = splade_encode(doc_text)
    return {"id": doc_id, "content": doc_text, "vector": vec}

def process_query(query_tuple):
    query_id, query_text = query_tuple
    vec = splade_encode(query_text)
    return {"id": query_id, "content": query_text, "vector": vec}

with concurrent.futures.ThreadPoolExecutor(max_workers=ths) as executor:
    docs_splade = list(executor.map(process_document, enumerate(documents)))

with concurrent.futures.ThreadPoolExecutor(max_workers=ths) as executor:
    query_splade = list(executor.map(process_query, enumerate(queries)))

with open("my_splade_docs.jsonl", "w", encoding="utf-8") as f:
    for obj in docs_splade:
        f.write(json.dumps(obj) + "\n")

with open("my_splade_queries.jsonl", "w", encoding="utf-8") as f:
    for obj in query_splade:
        f.write(json.dumps(obj) + "\n")
