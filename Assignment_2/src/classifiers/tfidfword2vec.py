from models.word2vec import Word2Vec
from utils.text import clean_texts
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from models.tfidf import TFIDF


class TFIDFWord2Vec(Word2Vec):
    def __init__(self):
        super().__init__()
        self.v2i = None
        self.tfidf = TFIDF()
        self.vocab = None

    def build_features(self, texts):
        if  self.v2i is None:
            tfidf_mat = self.tfidf.fit_transform(texts)
            self.v2i = {w: i for i, w in enumerate(self.tfidf.vocab)}
        else:
            tfidf_mat = self.tfidf.transform(texts)

        feats = []
        for row_idx, tokens in enumerate(texts):
            vecs, weights = [], []
            for tok in tokens:
                col = self.v2i.get(tok)
                if col is not None and tok in self.embedding:
                    vecs.append(self.embedding[tok])
                    weights.append(tfidf_mat[row_idx, col])

            if vecs:
                vecs  = np.vstack(vecs)
                w_arr = np.asarray(weights)
                w_sum = w_arr.sum()
                feats.append((vecs * w_arr[:, None]).sum(axis=0) / w_sum
                             if w_sum else vecs.mean(axis=0))
            else:
                feats.append(np.zeros(self.embedding.vector_size))

        return np.vstack(feats)