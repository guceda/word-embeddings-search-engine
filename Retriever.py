from rank_bm25 import BM25Okapi as BM25


class Retriever(object):
    model = 'BM25Okapi'

    def __init__(self, documents):
        self.corpus = documents
        self.bm25 = BM25(self.corpus)

    def query(self, tokenized_query, n=100):
        scores = self.bm25.get_scores(tokenized_query)
        best_docs = sorted(range(len(scores)), key=lambda i: -scores[i])[:n]
        return best_docs, [scores[i] for i in best_docs]
