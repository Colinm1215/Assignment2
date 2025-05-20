from classifiers.word2vec import Word2Vec
from utils.text import clean_texts
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from classifiers.tfidf import TFIDF


class TFIDFWord2Vec(Word2Vec):
    def __init__(self):
        super().__init__()
        self.tfidf = TFIDF()
        self.vocab = None

    def build_features(self, texts):
        self.tfidf.vocab = self.tfidf.build_vocab(texts)
        vocab_list = list(self.tfidf.vocab.keys())
        tf_matrix = self.tfidf.compute_tf(texts)
        idf_vector = self.tfidf.compute_idf(texts)
        tfidf_matrix = tf_matrix * idf_vector

        feats = []
        for idx, tokens in enumerate(texts):
            vecs, weights = [], []
            for token in tokens:
                if token in self.embedding:
                    vecs.append(self.embedding[token])
                    weights.append(tfidf_matrix[idx, vocab_list.index(token)] if token in vocab_list else 0)
            if vecs:
                vecs = np.vstack(vecs)
                w_arr = np.array(weights)
                if w_arr.sum() > 0:
                    feats.append((vecs * w_arr[:, None]).sum(axis=0) / w_arr.sum())
                else:
                    feats.append(vecs.mean(axis=0))
            else:
                feats.append(np.zeros(self.embedding.vector_size))

        return np.vstack(feats)