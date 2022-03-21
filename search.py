from gensim.models.keyedvectors import KeyedVectors
from utils import show_scores, tokenize
from Retriever import Retriever
from Ranker import Ranker
from utils import tokenize
import click
import pandas as pd
import gensim
import gensim.downloader as api
from gensim.models.word2vec import Word2Vec
import numpy as np
import logging
logging.basicConfig(level=logging.DEBUG)


@click.command()
@click.option("--query", prompt="Search query", help="Search query")
def main(query):
    print('Query: "{}"'.format(query))

    print("Reading documents...", end="")
    # documents = [
    #     "Distance between Madrid and Barcelona",
    #     "Who governs a country's airspace?",
    #     "What is a supermoon, and how noticeable is it to the naked eye?",
    #     "What the evidence says about police body-cameras",
    #     "Who controls Syria?",
    #     "Putin is invading Ucraine",
    #     "Cast float to int",
    # ]
    documents = [
        'Creates a new column that returns the arc tangent of the values of a numeric field. When applied with two arguments, it returns the arc tangent of the specified x- and y-coordinates.',
        'Creates a new column that shifts to the right the bits of the values in the first argument as many positions as specified in the second argument. This operation always fills vacant places after shifting with zeros, so the sign of the original number may vary. Use Bitwise right shift (rshift, >>) if you want to preserve the sign of the original number.',
        'Checks for the presence of one or more values in a given string. The filter will identify those strings containing at least one of the indicated values. Create column - Creates a Boolean column that shows true when at least one of the indicated values is present in the given string. If you enter your query using LINQ, note that the -> operator syntax does not admit more than two arguments. Use the has() syntax if you need to add more than one value. This operation is case sensitive. Use the Contains - case insensitive (weakhas) operation if you need to apply this filter ignoring case.',
        'Returns only those strings that contain a specified value, ignoring case. Create column - Creates a Boolean column that shows true when the indicated value is present in the given string, ignoring case. Use the Contains (has, ->) operation if you need to discriminate between uppercase and lowercase letters.',
        'Filter - Retrieves only absolute URIs from a specified field. Create column - Creates a Boolean column that shows true if a given URI is absolute.',
        'Create column - Creates a new column that returns the Damerau distance between two strings.',
        'Create column - Creates a new column that returns the Hamming distance between two strings.',
        'Create column - Creates a new column that returns the Levenshtein distance between two strings.',
        'Create column - Creates a new column that returns the Osa distance between two strings.',
    ]
    print(" [DONE]")
    print("Tokening documents...", end="")
    corpus = [list(gensim.utils.tokenize(doc.lower())) for doc in documents]
    tokenized_query = tokenize(query)
    print(" [DONE]")

    retriever = Retriever(corpus)
    retrieval_indexes, retrieval_scores = retriever.query(tokenized_query)

    retrieved_documents = [documents[idx] for idx in retrieval_indexes]
    print("======== BM25 ========")
    show_scores(retrieved_documents, retrieval_scores, 5)

    tokenzed_retrieved_documents = [corpus[idx] for idx in retrieval_indexes]
    retrieved_documents = corpus
    tokenzed_retrieved_documents = corpus

    print("Loading corpus...")

    # https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#:~:text=Word2Vec%20is%20a%20more%20recent,each%20other%20have%20differing%20meanings.
    #query_embedding = api.load('glove-wiki-gigaword-50')
    query_embedding = KeyedVectors.load_word2vec_format(
        "./models/SO_vectors_200.bin", binary=True)
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
