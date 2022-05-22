import numpy as np
from timeit import default_timer as timer

class Ranker(object):
    model = 'gsim'

    def __init__(self, query_embedding, document_embedding):
        self.query_embedding = query_embedding
        self.document_embedding = document_embedding

    def _create_mean_embedding(self, word_embeddings):
        return np.mean(word_embeddings, axis=0)

    def _create_max_embedding(self, word_embeddings):
        return np.amax(word_embeddings, axis=0)

    def _embed(self, tokens, embedding):
        word_embeddings = np.array([embedding[token]
                                   for token in tokens if token in embedding])
        mean_embedding = self._create_mean_embedding(word_embeddings)
        max_embedding = self._create_max_embedding(word_embeddings)
        embedding = np.concatenate([mean_embedding, max_embedding])
        unit_embedding = embedding / (embedding**2).sum()**0.5
        return unit_embedding

    def rank(self, tokenized_query, tokenized_documents):
        """
        Re-ranks a set of documents according to embedding distance
        """
        t1 = timer()
        query_embedding = self._embed(
            tokenized_query, self.query_embedding)  # (E,)
        document_embeddings = np.array([self._embed(
            document, self.document_embedding) for document in tokenized_documents])  # (N, E)
        scores = document_embeddings.dot(query_embedding)
        index_rankings = np.argsort(scores)[::-1]
        t2 = timer()
        print(f'Embeddings took {t2-t1}ms')
        return index_rankings, np.sort(scores)[::-1]
