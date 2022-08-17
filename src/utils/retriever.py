from typing import List
from typing import Tuple
from transformers import AutoTokenizer, AutoModel, AutoConfig
import numpy as np
import torch
from torch.nn.functional import normalize
import pickle
import json
from tqdm import tqdm

tokenizer_labse = AutoTokenizer.from_pretrained("setu4993/LaBSE")
model_labse = AutoModel.from_pretrained("setu4993/LaBSE")
words2IDF = {}
title_to_embeddings = {}
title_to_domain = {}
N_DOC = 488


def load_docs(path2doc) -> None:
    global title_to_embeddings, title_to_domain, words2IDF, N_DOC

    with open(path2doc, 'r') as f:
        multidoc2dial_doc = json.load(f)
    
    doc_title_train = []
    doc_texts_train = []
    for domain in multidoc2dial_doc['doc_data']:
        for title in multidoc2dial_doc['doc_data'][domain]:
            doc_title_train.append(title)
            doc_texts_train.append(multidoc2dial_doc['doc_data'][domain]\
                                          [title]['doc_text'].strip())
            title_to_domain[title] = domain
    titles = list(set(doc_title_train))
    N_DOC = len(titles)
    
    TRAIN_SIZE = len(titles)
    for title in tqdm(titles, desc="[Loading documents - title embedding]"):
        title_to_embeddings[title] = get_embeddings(title)
    
    words = set()
    doc_texts_train_tokenized = []
    for doc in doc_texts_train:
        tokenized_doc = [s.lower() for s in tokenizer_labse.tokenize(doc)]
        doc_texts_train_tokenized.append(tokenized_doc) 
        words = set(tokenized_doc).union(words)
    
    for word in tqdm(words, desc="[Loading documents - IDF scores]"):
        n_word = 0
        for doc in doc_texts_train_tokenized:
            if word in doc:
                n_word += 1
        words2IDF[word] = np.log(N_DOC / (n_word + 1))


def get_embeddings(sentece):
    """
    Return embeddings based on encoder model

    :param sentence: input sentence(s)
    :type sentence: str or list of strs
    :return: embeddings
    """
    tokenized = tokenizer_labse(sentece,
                                return_tensors="pt",
                                padding=True)
    with torch.no_grad():
        embeddings = model_labse(**tokenized)
    
    return np.squeeze(np.array(embeddings.pooler_output))


def calc_idf_score(sentence) -> float:
    """
    Calculate the mean idf score for given sentence.

    :param sentence: input sentence
    :type sentence: str
    :return: mean idf score of sentence token
    """
    tokenzied_sentence = [s.lower() for s in tokenizer_labse.tokenize(sentence)]
    score = 0
    for token in tokenzied_sentence:
        if token in words2IDF:
            score += words2IDF[token]
        else:
            score += np.log(N_DOC)
    return score / len(tokenzied_sentence)


def predict_labelwise_doc_at_history_ordered(queries, title_embeddings, k=1) -> Tuple[List[float], List[str]]:
    """
    Predict which document is matched to the given query.

    :param queries: input queries in time reversed order (latest first)
    :type queries: str (or list[str])
    :param title_embeddings: list of title embeddings
    :type title_embeddings: list[str]
    :param k: number of returning docs
    :type k: int 
    :return: return the document names and accuracies
    """
    similarities = np.array(list(map(lambda x: 0.0, title_embeddings)))
    coef_sum = 0
    for i, query in enumerate(queries):
        query_embd = get_embeddings(query)
        query_sim = list(map(lambda x: np.dot(x, query_embd) /
                            (np.linalg.norm(query_embd) * np.linalg.norm(x)),
                            title_embeddings))
        query_sim = np.array(query_sim)

        coef = 2**(-i) * calc_idf_score(query)
        coef_sum += coef
        similarities += coef * query_sim

    similarities = similarities / coef_sum
    best_k_idx = similarities.argsort()[::-1][:k]
    accuracy = similarities[best_k_idx]
    return (accuracy, best_k_idx)


def get_documents(domain, queries, k=10) -> List[str]:
    "returns list of related document IDs"
    titles = [title for title in title_to_embeddings.keys() if title_to_domain[title] == domain]
    title_embeddings = [title_to_embeddings[title] for title in titles]
    acc, best_k_idx = predict_labelwise_doc_at_history_ordered(queries, title_embeddings, k)
    return [titles[i] for i in best_k_idx]
