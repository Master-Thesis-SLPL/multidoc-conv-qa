"Bert for sequence classification."

import json, os
import pandas as pd

os.environ["WANDB_DISABLED"] = "true"

docs_path = '../../../dataset/multidoc2dial/v1.0/multidoc2dial_doc.json'
train_path = '../../../dataset/multidoc2dial/v1.0/multidoc2dial_dial_train.json'
eval_path = '../../../dataset/multidoc2dial/v1.0/multidoc2dial_dial_validation.json'

train_dataset_list = []
eval_dataset_list = []


with open(docs_path, 'r') as f:
    docs_data = json.load(f)['doc_data']

doc_names = [doc_id for domain, domain_docs in docs_data.items() for doc_id in domain_docs]

def make_spans_dataset_list():
    docs_dataset_list = []
    for domain, domain_docs in docs_data.items():
        for doc_id, doc_data in domain_docs.items():
            for span_id, span in doc_data['spans'].items():
                docs_dataset_list.append({
                    'query': span['text_sp'],
                    'label': doc_names.index(doc_id)
                })
    return docs_dataset_list


def make_sentences_dataset_list():
    import nltk
    import nltk.data
    nltk.download('punkt')
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    docs_dataset_list = []
    for domain, domain_docs in docs_data.items():
        for doc_id, doc_data in domain_docs.items():
            paragraphs = doc_data['doc_text'].split('\n')
            sentences = []
            for paragraph in paragraphs:
                sentences += tokenizer.tokenize(paragraph)
                
            sentences_3 = [sentences[i: i+4] for i in range(0, len(sentences), 4)]
            sentences = [" ".join(sentence) for sentence in sentences_3]
            
            for sentence in sentences:
                docs_dataset_list.append({
                    'query': sentence,
                    'label': doc_names.index(doc_id)
                })
                
    return docs_dataset_list


def make_utterances_dataset_list(dial_data_path):
    questions_dataset_list = []

    with open(dial_data_path, 'r') as f:
        dials_data = json.load(f)['dial_data']

    for domain, domain_dials in dials_data.items():
        for dial in domain_dials:
            for i, turn in enumerate(dial['turns']):
                query = dial['turns'][i]['utterance']
                doc_id = dial['turns'][i]['references'][0]['doc_id']
                questions_dataset_list.append({
                    'query': query,
                    'label': doc_names.index(doc_id)
                })

    return questions_dataset_list


def construct_history_dataset_list(dial_data_path):
    "retriever_dataset[query_text | title | gold_passage]"
    questions_dataset_list = []

    with open(dial_data_path, 'r') as f:
        dials_data = json.load(f)['dial_data']

    for domain, domain_dials in dials_data.items():
        for dial in domain_dials:
            prev_doc_id = ""
            current_query = ""
            for i, turn in enumerate(dial['turns']):
                doc_id = dial['turns'][i]['references'][0]['doc_id']
                if doc_id == prev_doc_id:
                    query = dial['turns'][i]['utterance']
                    current_query += query
                else:
                    query = dial['turns'][i]['utterance']
                    current_query = query
                    
                questions_dataset_list.append({
                    'query': current_query,
                    'label': doc_names.index(doc_id)
                })

    return questions_dataset_list


def construct_questions_dataset_list(dial_data_path):
    "retriever_dataset[query_text | title | gold_passage]"
    questions_dataset_list = []

    with open(dial_data_path, 'r') as f:
        questions_dataset = json.load(f)['dial_data']
    
    for domain, domain_dials in questions_dataset.items():
        for dial in domain_dials:
            for i, turn in enumerate(dial['turns'][:-1]):
                if turn['role'] == 'user':
                    if dial['turns'][i+1]['role'] == 'agent':
                        agent_turn = dial['turns'][i+1]
                        query = dial['turns'][i]['utterance']
                        doc_id = agent_turn['references'][0]['doc_id']
                        questions_dataset_list.append({
                            'query': query,
                            'label': doc_names.index(doc_id)
                        })
                    else:
                        continue
    return questions_dataset_list


# spans_dataset_list = make_spans_dataset_list()
spans_dataset_list = make_sentences_dataset_list()
print(spans_dataset_list[:4])
utterances_dataset_list = construct_history_dataset_list(train_path)
# utterances_dataset_list = make_utterances_dataset_list(train_path)

train_dataset_list = spans_dataset_list + utterances_dataset_list
eval_dataset_list = construct_history_dataset_list(eval_path)
# eval_dataset_list = construct_questions_dataset_list(eval_path)

train_df = pd.DataFrame(train_dataset_list)
test_df = pd.DataFrame(eval_dataset_list)

from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification
model_name = "setu4993/LaBSE"
tokenizer = AutoTokenizer.from_pretrained(model_name)

def tokenize_function(examples, prediction=False, cuda=False):
    if prediction:
        tokenized = tokenizer(examples['query'], max_length=256, padding="max_length", truncation=True, return_tensors='pt')
    else:
        tokenized = tokenizer(examples['query'], max_length=256, padding="max_length", truncation=True)
    if cuda:
        tokenized_cuda = {}
        for key, value in tokenized.items():
            tokenized_cuda[key] = value.cuda()
        return tokenized_cuda
    else:
        return tokenized


import json
import pandas as pd

from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer, AutoModel, AutoConfig, AutoModelForSequenceClassification
import numpy as np
import torch
from torch.nn.functional import normalize
from tqdm import tqdm
from datasets import load_metric



from datasets import Dataset, DatasetDict

train_dataset = Dataset.from_pandas(train_df)
test_dataset = Dataset.from_pandas(test_df)

tokenized_trainset = train_dataset.map(tokenize_function, batched=True)
tokenized_testset = test_dataset.map(tokenize_function, batched=True)

# tokenized_trainset = tokenized_trainset.rename_column("query", "label")
# tokenized_testset = tokenized_testset.rename_column("query", "label")

fud_dataset = DatasetDict()

fud_dataset['train'] = tokenized_trainset
fud_dataset['validation'] = tokenized_testset

model_name = "setu4993/LaBSE"
tokenizer = AutoTokenizer.from_pretrained(model_name)
fudnet_model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=len(doc_names))

device = torch.device("cuda:0")
fudnet_model.to(device)


import numpy as np
from datasets import load_metric

metric = load_metric("accuracy")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


from transformers import Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir='outputs_hf/',
    #cache_dir='outputs_hf_cache/',
    num_train_epochs=3,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    warmup_steps=50,
    weight_decay=0.01,
    logging_steps=100,
    evaluation_strategy='epoch',
    save_strategy ='epoch',
    load_best_model_at_end=True,
    # auto_find_batch_size=True,
)

trainer = Trainer(
    model=fudnet_model,
    args=training_args,
    train_dataset=fud_dataset['train'],
    eval_dataset=fud_dataset['validation'],
    compute_metrics=compute_metrics
)

trainer.train()
