"""
Python module presenting all diferent retrievers implemented
 in jupyter notebooks in a modular manner.
"""

import json
import os
import pickle
from typing import List, Tuple

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer, AutoModelForSequenceClassification


class DrTeitRetriever:
    file_title_embeddings = os.path.join('..', '..', 'retriever', 'cache', 'doc_title_LaBSE_Embedding.npy')
    file_title_to_embeddings = os.path.join('..', '..', 'retriever', 'cache', 'title_to_embeddings.pkl')
    file_words2IDF = os.path.join('..', '..', 'retriever', 'cache', 'words_to_IDF.pkl')

    def __init__(self):    
        self.tokenizer_labse = AutoTokenizer.from_pretrained("setu4993/LaBSE")
        self.model_labse = AutoModel.from_pretrained("setu4993/LaBSE")
        self.words2IDF = {}
        self.title_to_embeddings = {}
        self.title_to_domain = {}
        self.tfidfVectorizer = None
        self.tfidf_wm = None
        self.N_DOC = 488

    def load_docs(self, path2doc):
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

        if not os.path.exists(self.file_title_to_embeddings):
            for title in tqdm(titles, desc="[Loading documents - title embedding]"):
                self.title_to_embeddings[title] = self.get_embeddings(title)

            with open(self.file_title_to_embeddings, 'wb') as f:
                pickle.dump(self.title_to_embeddings, f)
        else:
            with open(self.file_title_to_embeddings, 'rb') as f:
                self.title_to_embeddings = pickle.load(f)

        self.tfidfVectorizer = TfidfVectorizer(strip_accents=None, analyzer='char', 
                        ngram_range=(2, 8), norm='l2', use_idf=True, smooth_idf=True)
        self.tfidf_wm = self.tfidfVectorizer.fit_transform(doc_texts_train)

        words = set()
        doc_texts_train_tokenized = []
        for doc in doc_texts_train:
            tokenized_doc = [s.lower() for s in self.tokenizer_labse.tokenize(doc)]
            doc_texts_train_tokenized.append(tokenized_doc) 
            words = set(tokenized_doc).union(words)
        

        if not os.path.exists(self.file_words2IDF):
            for word in tqdm(words, desc="[Loading documents - IDF scores]"):
                n_word = 0
                for doc in doc_texts_train_tokenized:
                    if word in doc:
                        n_word += 1
                self.words2IDF[word] = np.log(self.N_DOC / (n_word + 1))    
            
            with open(self.file_words2IDF, 'wb') as f:
                pickle.dump(self.words2IDF, f)
        else:
            with open(self.file_words2IDF, 'rb') as f:
                self.words2IDF = pickle.load(f)

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

    def calc_idf_score(self, sentence):
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

    def predict_dr_teit(self, queries, title_embeddings, k=1, alpha=10, coef_reverse_power=True):
        """
        Predict which document is matched to the given query.

        :param queries: input queries in time reversed order (latest first)
        :type queries: str (or list[str])
        :param title_embeddings: list of title embeddings
        :type title_embeddings: list[str]
        :param k: number of returning docs
        :type k: int 
        :param coef_reverse_power: use reverse 2^ power for the previous coefs or not
        :type coef_reverse_power: bool
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
            if coef_reverse_power:
                coef = 2**(-i) * self.calc_idf_score(query)
            else:
                coef = self.calc_idf_score(query)
            coef_sum += coef

            idf_score += coef * query_sim
            tfidf_score += coef * np.squeeze(np.asarray(self.tfidf_wm @ self.tfidfVectorizer.transform([query]).todense().T))

        scores = (idf_score + alpha * tfidf_score) / coef_sum
        best_k_idx = scores.argsort()[::-1][:k]
        scores = scores[best_k_idx]
        return (scores, best_k_idx)

    def get_documents(self, domain, queries, k=1):
        "returns list of related document IDs"
        if domain:
            titles = [title for title in self.title_to_embeddings.keys() if self.title_to_domain[title] == domain]
        else:
            titles = [title for title in self.title_to_embeddings.keys()]
        title_embeddings = [self.title_to_embeddings[title] for title in titles]
        acc, best_k_idx = self.predict_dr_teit(queries, title_embeddings, k)
        if domain:
            return [titles[i] for i in best_k_idx]
        else:
            return [(self.title_to_domain[titles[i]], titles[i]) for i in best_k_idx]


class DrFudRetriever(DrTeitRetriever):
    fudnet_model_name = "alistvt/fudnet"
    separation_token = " <SEP> "
    
    def __init__(self):
        super().__init__()
        self.tokenizer_fudnet = self.tokenizer_labse
        self.model_fudnet = AutoModelForSequenceClassification.from_pretrained(self.fudnet_model_name)
        device = torch.device("cuda:0")
        self.model_fudnet.to(device)

    def combine_and_tokenize(self, prev_question, current_question, prediction=False, cuda=True):
        combined = f"{prev_question}{self.separation_token}{current_question}"
        if prediction:
            tokenized = self.tokenizer_fudnet(combined, max_length=128, padding="max_length", truncation=True, return_tensors='pt')
        else:
            tokenized = self.tokenizer_fudnet(combined, max_length=128, padding="max_length", truncation=True)
        if cuda:
            tokenized_cuda = {}
            for key, value in tokenized.items():
                tokenized_cuda[key] = value.cuda()
            return tokenized_cuda
        else:
            return tokenized

    def get_documents(self, domain, queries, k=1):
        "returns list of related document IDs"
        if domain:
            titles = [title for title in self.title_to_embeddings.keys() if self.title_to_domain[title] == domain]
        else:
            titles = [title for title in self.title_to_embeddings.keys()]
        title_embeddings = [self.title_to_embeddings[title] for title in titles]

        if len(queries) == 1:
            prev_question, current_question = "", queries[0]
        else:
            prev_question, current_question = queries[2], queries[0]

        inputs = self.combine_and_tokenize(prev_question, current_question, prediction=True, cuda=True)
        outputs = self.model_fudnet(**inputs)
        is_followup = bool(torch.argmax(outputs.logits))
        
        if is_followup:
            dr_scores, dr_predictions = self.predict_dr_teit(queries[:3], title_embeddings, k, coef_reverse_power=False)
            return dr_predictions
        else:
            dr_scores, dr_predictions = self.predict_dr_teit([queries[0]], title_embeddings, k, coef_reverse_power=False)
            return dr_predictions
        
        if domain:
            return [titles[i] for i in best_k_idx]
        else:
            return [(self.title_to_domain[titles[i]], titles[i]) for i in best_k_idx]
