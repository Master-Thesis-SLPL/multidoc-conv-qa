import logging
import json
import pandas as pd

from simpletransformers.retrieval import RetrievalModel, RetrievalArgs

logging.basicConfig(level=logging.INFO)
transformers_logger = logging.getLogger("transformers")
transformers_logger.setLevel(logging.WARNING)


def construct_retriever_dataset(dial_path, docs_path):
    "retriever_dataset[query_text | title | gold_passage]"
    retriever_dataset = []

    with open(dial_path, 'r') as f:
        questions_dataset = json.load(f)['dial_data']
    
    with open(docs_path, 'r') as f:
        docs_dataset = json.load(f)['doc_data']

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

                            retriever_dataset.append({
                                'query_text': question,
                                'title': domain + ' <SEP> ' + docs_dataset[domain][doc_id]['title'],
                                'gold_passage': docs_dataset[domain][doc_id]['doc_text']
                            })
                        else:
                            # TODO: unknown
                            pass
                    else:
                        continue
    return retriever_dataset


train_data = construct_retriever_dataset(
    '../../../dataset/multidoc2dial/v1.0/multidoc2dial_dial_train.json',
    '../../../dataset/multidoc2dial/v1.0/multidoc2dial_doc.json'
)

eval_data = construct_retriever_dataset(
    '../../../dataset/multidoc2dial/v1.0/multidoc2dial_dial_validation.json',
    '../../../dataset/multidoc2dial/v1.0/multidoc2dial_doc.json'
)

train_df = pd.DataFrame(train_data)

eval_df = pd.DataFrame(eval_data)

# Configure the model
model_args = RetrievalArgs()
model_args.num_train_epochs = 40
model_args.no_save = True

model_type = "dpr"
context_encoder_name = "facebook/dpr-ctx_encoder-single-nq-base"
question_encoder_name = "facebook/dpr-question_encoder-single-nq-base"

model = RetrievalModel(
    model_type=model_type,
    context_encoder_name=context_encoder_name,
    query_encoder_name=question_encoder_name,
    from_tf=True
)

# Train the model
model.train_model(train_df, eval_data=eval_df)

# Evaluate the model
result = model.eval_model(eval_df)