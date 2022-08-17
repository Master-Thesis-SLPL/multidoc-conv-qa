import json
from tqdm import tqdm
import numpy as np

with open("doc2dial_doc.json", 'r') as f:
    docs = json.load(f)

docs = docs["doc_data"]


class Document:
    def __init__(self, doc_dict) -> None:
        self.spans = doc_dict["spans"]
        self.sections = {}
        for _, span in self.spans.items():
            sec_id = span["id_sec"]
            if sec_id in self.sections:
                self.sections[sec_id].append(span["id_sp"])
            else:
                self.sections[sec_id] = [span["id_sp"]]
        self.spans_count = len(self.spans)

    def get_stats(self):
        section_spans_count = [len(value) for key, value in self.sections.items()]
        return len(self.sections), self.spans_count, section_spans_count


section_span_counts = []
span_counts = []
section_counts = []

for domain, domain_docs in docs.items():
    for name, doc  in tqdm(domain_docs.items()):
        d = Document(doc)
        sec_count, sp_count, sec_sp_count = d.get_stats()
        section_span_counts += sec_sp_count
        span_counts.append(sp_count)
        section_counts.append(sec_count)


# for i, x in enumerate(section_span_counts):
#     if i%30 == 0:
#         end = "\n"
#     else:
#         end = " "
#     print(x, end=end)

from collections import Counter

c = Counter(section_span_counts)

print(c)
import matplotlib.pyplot as plt

plot = plt.bar(c.keys(), c.values())
plt.show()


section_spans_mean = np.mean(section_span_counts)
section_spans_std = np.std(section_span_counts)
section_spans_max = np.max(section_span_counts)
section_spans_min = np.min(section_span_counts)

spans_count_mean = np.mean(span_counts)
spans_count_std = np.std(span_counts)
spans_count_max = np.max(span_counts)
spans_count_min = np.min(span_counts)

sections_count_mean = np.mean(section_counts)
sections_count_std = np.std(section_counts)
sections_count_max = np.max(section_counts)
sections_count_min = np.min(section_counts)

print(
    f"""
section spans:
   mean: {section_spans_mean:.3f}
   std:  {section_spans_std:.3f}
   max:  {section_spans_max}
   min:  {section_spans_min}

sections count:
   mean: {sections_count_mean:.3f}
   std:  {sections_count_std:.3f}
   max:  {sections_count_max}
   min:  {sections_count_min}

spans count:
   mean: {spans_count_mean:.3f}
   std:  {spans_count_std:.3f}
   max:  {spans_count_max}
   min:  {spans_count_min}
    """
)