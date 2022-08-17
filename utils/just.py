

import json

with open("predictions.json", 'r') as f:
    data = f.read()

data = json.loads(data)

my_list = []
for id_, val in data.items():
    my_list.append({
        "id": id_,
        "grounding": val,
        "utterance": val
    })

json_obj = json.dumps(my_list, indent=2)

with open("predicted.json", "w", encoding="utf-8") as f:
    f.write(json_obj)