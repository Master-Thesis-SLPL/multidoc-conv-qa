BASE_DIR = '/hdd2/alistvt/multidoc/multidoc-conv-qa'

import json
import pandas as pd

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification
import numpy as np
import torch
from torch.nn.functional import normalize
from tqdm import tqdm
from datasets import load_metric

def construct_sections_dict(filepath):
    "sections[domain][doc_id][sec_id]['text'/'title'/'spans: list']"
    sections = {}
    import json
    with open(filepath, 'r') as f:
        multidoc2dial_docs = json.load(f)['doc_data']
    
    for domain in multidoc2dial_docs:
        sections[domain] = {}

    for domain, domain_docs in multidoc2dial_docs.items():
        for doc_id in domain_docs:
            sections[domain][doc_id] = {}
    
    for domain, domain_docs in multidoc2dial_docs.items():
        for doc_id, doc_data in domain_docs.items():
            for span_id, span_data in doc_data['spans'].items():
                id_sec = span_data['id_sec']
                text_sec = span_data['text_sec'].strip()
                title_sec = span_data['title'].strip()
                if id_sec not in sections[domain][doc_id]:
                    sections[domain][doc_id][id_sec] = {
                        'title': title_sec,
                        'text': text_sec,
                        'spans': [span_id]
                    }
                else:
                    sections[domain][doc_id][id_sec]['spans'].append(span_id)

    return sections    


sections_dict = construct_sections_dict(BASE_DIR + '/dataset/multidoc2dial/v1.0/multidoc2dial_doc.json')

def construct_sections_dataset(filepath, sections_dict):
    "sections_dataset[question | section_title | section_text | label]"
    sections_dataset = []
    with open(filepath, 'r') as f:
        questions_dataset = json.load(f)['dial_data']
    
    for domain, domain_dials in questions_dataset.items():
        for dial in domain_dials:
            for i, turn in enumerate(dial['turns'][:-1]):
                if turn['role'] == 'user':
                    if dial['turns'][i+1]['role'] == 'agent':
                        agent_turn = dial['turns'][i+1]
                        question = dial['turns'][i]['utterance']
                        if len(agent_turn['references']):
                            reference = agent_turn['references'][0]
                            doc_id = reference['doc_id']
                            span_id = reference['id_sp']

                            for section_id, section in sections_dict[domain][doc_id].items():
                                if span_id in section['spans']:
                                    label = True
                                else:
                                    label = False
                                sections_dataset.append({
                                    'question': question,
                                    'section_title': section['title'],
                                    'section_text': section['text'],
                                    'label': label
                                })
                        else:
                            # TODO: unknown
                            pass
                    else:
                        continue
    return sections_dataset


train_dataset = construct_sections_dataset(BASE_DIR + '/dataset/multidoc2dial/v1.0/multidoc2dial_dial_train.json', sections_dict)
test_dataset = construct_sections_dataset(BASE_DIR + '/dataset/multidoc2dial/v1.0/multidoc2dial_dial_validation.json', sections_dict)

train_df = pd.DataFrame(train_dataset)
test_df = pd.DataFrame(test_dataset)


train_df_false = train_df[train_df["label"] == False]
train_df_true = train_df[train_df["label"] == True]

test_df_false = test_df[test_df["label"] == False]
test_df_true = test_df[test_df["label"] == True]

print(train_df_true.shape)
print(train_df_false.shape)
print(test_df_true.shape)
print(test_df_false.shape)


from sklearn.utils import resample

train_df_false_down = resample(train_df_false,
             replace=True,
             n_samples=3*len(train_df_true),
             random_state=42)

test_df_false_down = resample(test_df_false,
             replace=True,
             n_samples=3*len(test_df_true),
             random_state=42)

print(train_df_false_down.shape)
print(test_df_false_down.shape)

train_df_downsampled = pd.concat([train_df_false_down, train_df_true])
test_df_downsampled = pd.concat([test_df_false_down, test_df_true])

print(train_df_downsampled.shape)
print(test_df_downsampled.shape)

train_df = train_df_downsampled
test_df = test_df_downsampled

model_name = "setu4993/LaBSE"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)


def combine_sample(sample, sep_token=" <SEP> "):
    return sample['question'] + sep_token + sample['section_title'] + sep_token + sample['section_text']


def tokenize_function(examples, prediction=False, cuda=False):
    if prediction:
        tokenized = tokenizer(examples['combined'], max_length=512, padding="max_length", truncation=True, return_tensors='pt')
    else:
        tokenized = tokenizer(examples['combined'], max_length=512, padding="max_length", truncation=True)
    if cuda:
        tokenized_cuda = {}
        for key, value in tokenized.items():
            tokenized_cuda[key] = value.cuda()
        return tokenized_cuda
    else:
        return tokenized

import pickle, os
secpred_ds_file = 'secpred_ds.pkl'

if not os.path.exists(secpred_ds_file):

    train_df['combined'] = train_df.apply(combine_sample, axis = 1)
    test_df['combined'] = test_df.apply(combine_sample, axis = 1)

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    tokenized_trainset = train_dataset.map(tokenize_function, batched=True)
    tokenized_testset = test_dataset.map(tokenize_function, batched=True)

    secpred_dataset = DatasetDict()

    secpred_dataset['train'] = tokenized_trainset
    secpred_dataset['validation'] = tokenized_testset
    
    with open(secpred_ds_file, 'wb') as f:
        pickle.dump(secpred_dataset, f)
else:
    with open(secpred_ds_file, 'rb') as f:
        secpred_dataset = pickle.load(f)



model_name = "setu4993/LaBSE"

secpred_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

device = torch.device("cuda:0")
secpred_model.to(device)


metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir=BASE_DIR + '/experiments/',
    num_train_epochs=3,
    per_device_train_batch_size=4,
    per_device_eval_batch_size=4,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=10,
    evaluation_strategy='epoch',
    save_strategy ='epoch',
    load_best_model_at_end=True,
    # auto_find_batch_size=True,
)

trainer = Trainer(
    model=secpred_model,
    args=training_args,
    train_dataset=secpred_dataset['train'],
    eval_dataset=secpred_dataset['validation'],
    compute_metrics=compute_metrics
)


trainer.train()

