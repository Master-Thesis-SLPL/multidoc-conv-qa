import json

with open ('doc2dial_dial_validation__real.json', encoding="utf-8") as f:
    data = json.load(f)

del data['dial_data']['dmv']
del data['dial_data']['studentaid']
del data['dial_data']['ssa']

with open('doc2dial_dial_validation.json', 'w') as outfile:
    json.dump(data, outfile)