import logging
import json
import pandas as pd
from tqdm import tqdm

from simpletransformers.retrieval import RetrievalModel, RetrievalArgs

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def get_answers_rc(references, doc):
        """Obtain the grounding annotation for evaluation of subtask1."""
        if not references:
            return []
        start, end = -1, -1
        ls_sp = []
        doc_id = None
        for ele in references:
            spans = doc["spans"]
            doc_text = doc["doc_text"]
            
            id_sp = ele["id_sp"]
            start_sp, end_sp = spans[id_sp]["start_sp"], spans[id_sp]["end_sp"]
            if start == -1 or start > start_sp:
                start = start_sp
            if end < end_sp:
                end = end_sp
            ls_sp.append(doc_text[start_sp:end_sp])
        answer = {
            "text": doc_text[start:end],
            "answer_start": start,
            # "spans": ls_sp
        }
        return answer


def get_section_title(references, doc):
    ref = references[0]
    spans = doc["spans"]
    doc_text = doc["doc_text"]
    id_sp = ref["id_sp"]
    return spans[id_sp]['title']


def get_document_title(references, doc):
    return doc['title']    


def construct_retriever_dataset(dial_path, docs_path, get_passage_method):
    "retriever_dataset[query_text | title | gold_passage]"
    retriever_dataset = []

    with open(dial_path, 'r') as f:
        questions_dataset = json.load(f)['dial_data']
    
    with open(docs_path, 'r') as f:
        docs_dataset = json.load(f)['doc_data']

    for domain, domain_dials in tqdm(questions_dataset.items(), 
            desc=f"generating train data with {get_passage_method.__name__}"):
        for dial in domain_dials:
            for i, turn in enumerate(dial['turns'][:-1]):
                if turn['role'] == 'user':
                    if dial['turns'][i+1]['role'] == 'agent':
                        agent_turn = dial['turns'][i+1]
                        question = dial['turns'][i]['utterance']
                        
                        doc_id = agent_turn['references'][0]['doc_id']
                        doc = docs_dataset[domain][doc_id]
                        gold_passage = get_passage_method(agent_turn['references'], doc)

                        retriever_dataset.append({
                            'query_text': question,
                            # 'title': domain + ' <SEP> ' + docs_dataset[domain][doc_id]['title'],
                            'title': docs_dataset[domain][doc_id]['title'],
                            'gold_passage': docs_dataset[domain][doc_id]['doc_text']
                        })
                    else:
                        continue
    return retriever_dataset

    
train_data_spans = construct_retriever_dataset(
    '../../../dataset/multidoc2dial/v1.0/multidoc2dial_dial_train.json',
    '../../../dataset/multidoc2dial/v1.0/multidoc2dial_doc.json',
    get_answers_rc
)
train_data_sections = construct_retriever_dataset(
    '../../../dataset/multidoc2dial/v1.0/multidoc2dial_dial_train.json',
    '../../../dataset/multidoc2dial/v1.0/multidoc2dial_doc.json',
    get_section_title
)
train_data_doc_titles = construct_retriever_dataset(
    '../../../dataset/multidoc2dial/v1.0/multidoc2dial_dial_train.json',
    '../../../dataset/multidoc2dial/v1.0/multidoc2dial_doc.json',
    get_document_title
)

train_data = train_data_spans + train_data_sections + train_data_doc_titles
eval_data = construct_retriever_dataset(
    '../../../dataset/multidoc2dial/v1.0/multidoc2dial_dial_validation.json',
    '../../../dataset/multidoc2dial/v1.0/multidoc2dial_doc.json',
    get_answers_rc
)

train_df = pd.DataFrame(train_data)
eval_df = pd.DataFrame(eval_data)

train_queries = [td['query_text'] for td in train_data]
train_passages = [td['gold_passage'] for td in train_data]

# print(eval_df.iloc[0])
# input("IS THIS OK? [y/n]")

# Configure the model
model_args = RetrievalArgs(
    # hard_negatives = True,
    num_train_epochs = 50,
    no_save = True,
    use_early_stopping = True,
    overwrite_output_dir = True,
    train_batch_size = 32,
    eval_batch_size = 64,

    evaluate_during_training_verbose = True,
    evaluate_during_training = True,
    evaluate_during_training_silent = False,
    evaluate_each_epoch = True,
)

model = RetrievalModel(
    model_type = "dpr",
    context_encoder_name = "facebook/dpr-ctx_encoder-single-nq-base",
    query_encoder_name = "facebook/dpr-question_encoder-single-nq-base",
    args = model_args
)

print("Building hard negatives")
# Hard negatives
hard_df = model.build_hard_negatives(
    queries = train_queries,
    passage_dataset = train_passages,
    retrieve_n_docs = 1
)

train_df = train_df.assign(hard_negative = hard_df.values.tolist())
print(train_df.head())

# Train the model
model.train_model(
    train_data = train_df, 
    eval_data = eval_df,
    verbose = True,
)

# Evaluate the model
result = model.eval_model(eval_df)