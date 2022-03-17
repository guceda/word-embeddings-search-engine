from asyncio.log import logger
from EngineStatus import Status
from Ranker import Ranker
from Retriever import Retriever
from utils import show_scores

from utils import tokenize
import gensim
import gensim.downloader as api

DATASET = 'glove-twitter-50'

# logging.basicConfig(level=logging.DEBUG)


class SearchEngine:
    def __init__(self, logger=print):
        self.__logger = logger
        self.__status = Status.DOWN
        self.__docs = None
        self.__tokenized_docs = None
        self.__query_embedding = None

    def load_query_embedding(self):
        self.__setStatus(Status.PREPARING)
        corpus = api.load(DATASET)
        self.__setStatus(Status.READY)
        self.__query_embedding = corpus

    def __prepare_docs(self, documents):
        self.__setStatus(Status.PREPARING)
        self.__logger(f"Tokenizing documents...")
        clean_documents = [document['text'] for document in documents]
        corpus = [list(gensim.utils.tokenize(doc.lower()))
                  for doc in clean_documents]
        self.__setStatus(Status.READY)
        return (documents, corpus)

    def __prepare_query(self, query):
        self.__logger(f"Tokenizing query...")
        return tokenize(query)

    def get_status(self):
        return self.__status

    def __setStatus(self, status: Status):
        self.__logger(f"MODEL STATUS: {status}")
        self.__status = status

    def train(self, docs):
        # self.__query_embedding = self.__load_query_embedding()
        self.__docs, self.__tokenized_docs = self.__prepare_docs(docs)

    def search(self, raw_query, dual=False):
        documents = self.__docs
        tokenized_documents = self.__tokenized_docs
        if dual:
            retrieved_documents, tokenzed_retrieved_documents, retrieval_scores = self.retrieve(
                raw_query)

            documents = retrieved_documents
            tokenized_documents = tokenzed_retrieved_documents

        results = self.rank(raw_query, documents, tokenized_documents)
        return {
            'data': results,
            'model': Retriever.model + (f' + {Ranker.model}' if dual else ''),
            'object': DATASET
        }

    def retrieve(self, raw_query, ):
        tokenized_query = self.__prepare_query(raw_query)
        retriever = Retriever(self.__tokenized_docs)
        retrieval_indexes, retrieval_scores = retriever.query(tokenized_query)

        retrieved_documents = [self.__docs[idx] for idx in retrieval_indexes]
        tokenized_retrieved_documents = [
            self.__tokenized_docs[idx] for idx in retrieval_indexes]

        print("======== BM25 ========")
        show_scores(retrieved_documents, retrieval_scores,
                    len(retrieved_documents))
        return (retrieved_documents, tokenized_retrieved_documents, retrieval_scores)

    def rank(self, raw_query, retrieved_documents, tokenized_retrieved_documents):
        self.__setStatus(Status.RANKING)
        tokenized_query = self.__prepare_query(raw_query)

        self.__logger(f"Ranking documents...")
        ranker = Ranker(query_embedding=self.__query_embedding,
                        document_embedding=self.__query_embedding)

        ranker_indexes, ranker_scores = ranker.rank(
            tokenized_query, tokenized_retrieved_documents)
        reranked_documents = [retrieved_documents[idx]
                              for idx in ranker_indexes]
        print(" [DONE]")
        print("======== Embedding ========")
        show_scores(reranked_documents, ranker_scores, len(reranked_documents))
        self.__setStatus(Status.READY)

        results = [{'object': DATASET,
                    'document': float(i),
                    'score': float(ranker_scores[i]),
                    'text': self.__docs[ranker_indexes[i]]['text'],
                    'metadata': self.__docs[ranker_indexes[i]]['metadata'],
                    } for i in range(len(ranker_indexes))]
        return results