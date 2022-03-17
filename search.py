import tokenize
import click
import pandas as pd
import gensim
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
import numpy as np
import logging

# logging.basicConfig(level=logging.DEBUG)

from Ranker import Ranker
from Retriever import Retriever
from utils import show_scores, tokenize


@click.command()
@click.option("--query", prompt="Search query", help="Search query")
def main(query):
    print('Query: "{}"'.format(query))

    print("Reading documents...", end="")
    documents = [
        "An investment bonanza is coming",
        "Who governs a country's airspace?",
        "What is a supermoon, and how noticeable is it to the naked eye?",
        "What the evidence says about police body-cameras",
        "Who controls Syria?",
        "Putin is invading Ucraine"
    ]
    print(" [DONE]")
    print("Tokening documents...", end="")
    corpus = [list(gensim.utils.tokenize(doc.lower())) for doc in documents]
    tokenized_query = tokenize(query)
    print(" [DONE]")

    # retriever = Retriever(corpus)
    # retrieval_indexes, retrieval_scores = retriever.query(tokenized_query)

    # retrieved_documents = [documents[idx] for idx in retrieval_indexes]
    # print("======== BM25 ========")
    # show_scores(retrieved_documents, retrieval_scores, 5)

    # tokenzed_retrieved_documents = [corpus[idx] for idx in retrieval_indexes]
    retrieved_documents = corpus
    tokenzed_retrieved_documents = corpus

    print("Loading corpus...")

    query_embedding = api.load('glove-wiki-gigaword-50')
    print("Ranking documents...", end="")
    ranker = Ranker(query_embedding=query_embedding,
                    document_embedding=query_embedding)
    ranker_indexes, ranker_scores = ranker.rank(
        tokenized_query, tokenzed_retrieved_documents)
    reranked_documents = [retrieved_documents[idx] for idx in ranker_indexes]
    print(" [DONE]")
    print("======== Embedding ========")
    show_scores(reranked_documents, ranker_scores, 5)

    # corpus = api.load('text8')
    # model = Word2Vec(corpus)  # train a model from the corpus
    # model.wv.most_similar("car")

if __name__ == "__main__":
    main()
