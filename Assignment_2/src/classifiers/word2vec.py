import numpy as np
import gensim.downloader as gensim_api
from utils.text import clean_texts
from sklearn.linear_model import LogisticRegression


class Word2Vec:
    def __init__(self):
        self.embedding = gensim_api.load("word2vec-google-news-300")
        self.model = LogisticRegression()

    def preprocess(self, texts):
        return [text.split() for text in clean_texts(texts)]

    def build_features(self, texts):
        return self._pool(texts)

    def _pool(self, tokenized_texts):
        feats = []
        for tokens in tokenized_texts:
            vecs = [self.embedding[w] for w in tokens if w in self.embedding]
            if vecs:
                feats.append(np.mean(vecs, axis=0))
            else:
                feats.append(np.zeros(self.embedding.vector_size))
        return np.vstack(feats)

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)
