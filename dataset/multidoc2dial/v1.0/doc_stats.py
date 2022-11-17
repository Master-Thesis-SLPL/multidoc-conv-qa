import pandas as pd
import json
from tqdm import tqdm
import numpy as np

# with open("multidoc2dial_dial_train.json", 'r') as f:
with open("multidoc2dial_doc.json", 'r') as f:
    docs = json.load(f)["doc_data"]


tokens_count = []
spans_count = []
sections_count = []


for domain, domain_docs in docs.items():
    for doc_title, doc in tqdm(domain_docs.items()):
        tokens_count.append( len(doc['doc_text'].split()))
        spans_count.append( len(doc['spans'].keys()))
        sections_count.append( int(list(doc['spans'].values())[-1]['id_sec'].split('_')[-1]))

        
df = pd.DataFrame({
    'tokens': tokens_count,
    'spans': spans_count,
    'sections': sections_count
})


print(df)
print(df.describe(percentiles=[.25, .5, .75, .9, .95]))