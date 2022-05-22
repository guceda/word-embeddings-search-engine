import numpy as np

class BM25:
    def __init__(self, documents, k=1.5, b=0.75):
        self.corpus = documents
        self.k = k
        self.b = b
        self.__avgd1 = self.__avgd1()

    def __avgd1(self):
        """
        Compute average length of all documents
        """
        return sum(len(sentence) for sentence in self.corpus) / len(self.corpus)

    def __compute_sum(self, word_embeddings):
        return np.sum(word_embeddings, axis=1)

    def bm25(self, word, sentence):
        """
        Compute BM25 improved TF-IDF.
        """
        N = len(self.corpus)
        # term frequency f(q,D)
        freq = sentence.count(word)
        tf = (freq * (self.k+1)) / (freq + self.k *
                               (1 - self.b + self.b * len(sentence) / self.__avgd1))
        # inverse document frequency
        # number of documents that contain the keyword
        N_q = sum([1 for doc in self.corpus if word in doc])
        idf = np.log(((N - N_q + 0.5) / (N_q + 0.5)) + 1)
        return tf*idf

    def rank(self, tokenized_query, limit=None):
        bm25_matrix = []

        for document_idx in range(len(self.corpus)):
            document = self.corpus[document_idx]
            for word in tokenized_query:
                if len(bm25_matrix) <= document_idx:
                    bm25_matrix.append([])
                bm25_matrix[document_idx].append(self.bm25(word, document))
                

        scores = self.__compute_sum(bm25_matrix)
        best_docs = sorted(range(len(scores)), key=lambda i: -scores[i])[:limit]
        return (best_docs, scores)

