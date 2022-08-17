import json
from tqdm import tqdm
import numpy as np

with open("doc2dial_dial_train.json", 'r') as f:
    docs = json.load(f)

docs = docs["dial_data"]


class Dialog:
    def __init__(self, doc_dict) -> None:
        self.turns = doc_dict["turns"]
        self.ref_counts = []
        for turn in self.turns:
            self.ref_counts.append(len(turn["references"]))

    def get_stats(self):
        return self.ref_counts


ref_counts = []

for domain, domain_docs in docs.items():
    for name, documents  in tqdm(domain_docs.items()):
        for doc in documents:
            d = Dialog(doc)
            ref_counts += d.get_stats()


from collections import Counter

c = Counter(ref_counts)

print(c)

import matplotlib.pyplot as plt

plot = plt.bar(c.keys(), c.values())
plt.show()


section_spans_mean = np.mean(ref_counts)
section_spans_std = np.std(ref_counts)
section_spans_max = np.max(ref_counts)
section_spans_min = np.min(ref_counts)

print(
    f"""
reference counts:
   mean: {section_spans_mean:.3f}
   std:  {section_spans_std:.3f}
   max:  {section_spans_max}
   min:  {section_spans_min}
    """
)