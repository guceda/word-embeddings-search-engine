# from rank_bm25 import BM25Okapi as BM25a
from timeit import default_timer as timer
from BM25 import BM25


class Retriever(object):
    """
    BM25 preprocessing. Initial screening to get the n most relevant entries.
    """
    model = 'BM25Okapi'

    def __init__(self, documents):
        self.corpus = documents
        self.bm25 = BM25(self.corpus)

    def query(self, tokenized_query, n=100):
        t1 = timer()
        indexes, scores = self.bm25.rank(tokenized_query, n)
        best_docs = sorted(range(len(scores)), key=lambda i: -(scores[i]))
        t2 = timer()
        print(f'BM25 took {t2-t1}ms')
        return best_docs, [scores[i] for i in best_docs]
