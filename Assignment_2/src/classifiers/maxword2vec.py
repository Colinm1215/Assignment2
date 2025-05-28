from models.word2vec import Word2Vec
import numpy as np


class MaxWord2Vec(Word2Vec):
    def _pool(self, tokenized_texts):
        feats = []
        for tokens in tokenized_texts:
            vecs = [self.embedding[w] for w in tokens if w in self.embedding]
            if vecs:
                feats.append(np.max(np.vstack(vecs), axis=0))
            else:
                feats.append(np.zeros(self.embedding.vector_size))
        return np.vstack(feats)