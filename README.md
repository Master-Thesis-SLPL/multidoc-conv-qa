# Docalog-2022

## Retriever


Docalog: Multi-document Dialogue System using Transformer-based Span Retrieval - MultiDoc2dia 2022 Challenge.

This repository contains notebooks which are used during this challenge. They're showing our efforts to making the best predictions on document retriever part so they would be useful as a practice pattern.


### IR_PLDA

This file contains our first efforts for making the document retriever which starts with using PLDA method. If you're looking for our last (best efforts) you can see the `DR_TEIT.ipynb` file

### DR_TEIT

This file contains some tested methods for document retriever which you can see them in below table. We called our best model "Document Retrival with Title Embedding and IDF on Texts (DR.TEIT)". In this method we used two scoring measure and aggregate them by a convex combination  (λ * Similiarity_{Title Embedding} + (1 - λ) * Similiarity_{TextIDF}).

We used LaBSE model for out embeddings. For computing title embedding similarities we used cosine similarity between query embeddings and each document's title embedding. For the second part we used character-level (2gram to 8gram). We also trained our TF-IDF transformation matrix on the Multidoc2dial2022 documnets.


<div align="center">
  
| Method | @1 | @5 | @10 | @50 | @100 | MRR (mean, var) |
|:------:|:------:|:------:|:-------:|:-------:|:--------:|:---:|
| IDF - vanilla | 13% | 30% | 39% | 64% | 83% | (0.22, 0.11) |
| IDF - power-order | 15% | 31% | 41% | 65% | 83% | (0.23, 0.12) |
| IDF - power-order (softmax) | 10.7% | 23% | 31% | 57.6% | 78% | (0.18, 0.09) |
| IDF - self-attention | 13.9% | 29% | 38% | 62% | 82% | (0.22, 0.11) |
| **DR. TEIT** | **61.6%** | **86%** | **91%** | **96%** | **98%** | **(0.72, 0.13)** |

</div>

## Span Prediction

The main model which stack the **document retriever** and **span predictor** part is in this repository ([CAiRE](https://github.com/sharif-multidoc2dial/CAiRE)).

## Span Picker

## Generation

