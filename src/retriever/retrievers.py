"""
Python module presenting all diferent retrievers implemented
 in jupyter notebooks in a modular manner.
"""

import json
import pickle
from typing import List
from typing import Tuple

import numpy as np
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig


class DrTeitRetriever:
    def __init__(self):    
        self.tokenizer_labse = AutoTokenizer.from_pretrained("setu4993/LaBSE")
        self.model_labse = AutoModel.from_pretrained("setu4993/LaBSE")
        self.words2IDF = {}
        self.title_to_embeddings = {}
        self.title_to_domain = {}
        self.tfidfVectorizer = None
        self.tfidf_wm = None
        self.N_DOC = 488

    def load_docs(self, path2doc) -> None:
        with open(path2doc, 'r') as f:
            multidoc2dial_doc = json.load(f)
        
        doc_title_train = []
        doc_texts_train = []
        for domain in multidoc2dial_doc['doc_data']:
            for title in multidoc2dial_doc['doc_data'][domain]:
                doc_title_train.append(title)
                doc_texts_train.append(multidoc2dial_doc['doc_data'][domain]\
                                            [title]['doc_text'].strip())
                self.title_to_domain[title] = domain
        titles = doc_title_train
        self.N_DOC = len(titles)
        
        TRAIN_SIZE = len(titles)
        for title in tqdm(titles, desc="[Loading documents - title embedding]"):
            self.title_to_embeddings[title] = self.get_embeddings(title)
        
        self.tfidfVectorizer = TfidfVectorizer(strip_accents=None,
                                        analyzer='char',
                                        ngram_range=(2, 8),
                                        norm='l2',
                                        use_idf=True,
                                        smooth_idf=True)
        self.tfidf_wm = self.tfidfVectorizer.fit_transform(doc_texts_train)

        words = set()
        doc_texts_train_tokenized = []
        for doc in doc_texts_train:
            tokenized_doc = [s.lower() for s in self.tokenizer_labse.tokenize(doc)]
            doc_texts_train_tokenized.append(tokenized_doc) 
            words = set(tokenized_doc).union(words)
        
        for word in tqdm(words, desc="[Loading documents - IDF scores]"):
            n_word = 0
            for doc in doc_texts_train_tokenized:
                if word in doc:
                    n_word += 1
            self.words2IDF[word] = np.log(self.N_DOC / (n_word + 1))

    def get_embeddings(self, sentece):
        """
        Return embeddings based on encoder model

        :param sentence: input sentence(s)
        :type sentence: str or list of strs
        :return: embeddings
        """
        tokenized = self.tokenizer_labse(sentece,
                                    return_tensors="pt",
                                    padding=True)
        with torch.no_grad():
            embeddings = self.model_labse(**tokenized)
        
        return np.squeeze(np.array(embeddings.pooler_output))

    def calc_idf_score(self, sentence) -> float:
        """
        Calculate the mean idf score for given sentence.

        :param sentence: input sentence
        :type sentence: str
        :return: mean idf score of sentence token
        """
        tokenzied_sentence = [s.lower() for s in self.tokenizer_labse.tokenize(sentence)]
        score = 0
        for token in tokenzied_sentence:
            if token in self.words2IDF:
                score += self.words2IDF[token]
            else:
                score += np.log(self.N_DOC)
        if len(tokenzied_sentence) == 0:
            return 0
        return score / len(tokenzied_sentence)

    def predict_labelwise_doc_at_history_ordered(self, queries, title_embeddings, k=1, alpha=10) -> Tuple[List[float], List[str]]:
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
        idf_score = np.array(list(map(lambda x: 0.0, title_embeddings)))
        tfidf_score = np.array(list(map(lambda x: 0.0, title_embeddings)))
        coef_sum = 0
        for i, query in enumerate(queries):
            query_embd = self.get_embeddings(query)
            query_sim = list(map(lambda x: np.dot(x, query_embd) /
                                (np.linalg.norm(query_embd) * np.linalg.norm(x)),
                                title_embeddings))
            query_sim = np.array(query_sim)
            coef = 2**(-i) * self.calc_idf_score(query)
            coef_sum += coef

            idf_score += coef * query_sim
            tfidf_score += coef * np.squeeze(np.asarray(self.tfidf_wm @ self.tfidfVectorizer.transform([query]).todense().T))

        scores = (idf_score + alpha * tfidf_score) / coef_sum
        best_k_idx = scores.argsort()[::-1][:k]
        scores = scores[best_k_idx]
        return (scores, best_k_idx)

    def get_documents(self, domain, queries, k=1) -> List[str]:
        "returns list of related document IDs"
        if domain:
            titles = [title for title in self.title_to_embeddings.keys() if self.title_to_domain[title] == domain]
        else:
            titles = [title for title in self.title_to_embeddings.keys()]
        title_embeddings = [self.title_to_embeddings[title] for title in titles]
        acc, best_k_idx = self.predict_labelwise_doc_at_history_ordered(queries, title_embeddings, k)
        if domain:
            return [titles[i] for i in best_k_idx]
        else:
            return [(self.title_to_domain[titles[i]], titles[i]) for i in best_k_idx]
