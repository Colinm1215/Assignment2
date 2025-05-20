import numpy as np
from collections import Counter
from utils.text import clean_texts


class TFIDF:
    def __init__(self, top_n=1000, n_iters=2000, learning_rate=0.01, lambda_param=0.0002, ngram_range=(1, 1)):
        self.top_n = top_n
        self.n_iters = n_iters
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.vocab = {}
        self.ngram_range = ngram_range

    def generate_ngrams(self, tokens):
        min_n, max_n = self.ngram_range
        ngrams = []
        length = len(tokens)
        for n in range(min_n, max_n + 1):
            for i in range(length - n + 1):
                ngram = "_".join(tokens[i : i + n])
                ngrams.append(ngram)
        return ngrams

    def preprocess(self, texts):
        cleaned = clean_texts(texts)
        tokenized = [text.split() for text in cleaned]
        return [self.generate_ngrams(tokens) for tokens in tokenized]

    def build_vocab(self, texts):

        word_counter = Counter()

        for text in texts:
            try:
                word_counter.update(text)
            except TypeError as e:
                print(f"TypeError: {e} : {text}")

        vocab = {}
        for word, count in word_counter.most_common(self.top_n):
            vocab.update({word: count})

        return vocab

    def compute_tf(self, texts):
        n_docs = len(texts)
        vocab_list = list(self.vocab.keys())
        tf_matrix = np.zeros((n_docs, len(vocab_list)))

        for i, text in enumerate(texts):
            tf_c = Counter(text)
            n = len(text)
            for (word, count) in tf_c.items():
                if word in self.vocab:
                    tf_matrix[i, vocab_list.index(word)] = count / n

        return tf_matrix

    def compute_idf(self, texts):
        n_docs = len(texts)
        doc_freq = np.zeros(len(self.vocab))
        vocab_list = list(self.vocab.keys())

        for text in texts:
            unique_words = set(text)
            for word in unique_words:
                if word in self.vocab:
                    doc_freq[vocab_list.index(word)] += 1

        idf_vector = np.log((n_docs + 1) / (doc_freq + 1)) + 1
        return idf_vector

    def build_features(self, texts):
        self.vocab = self.build_vocab(texts)
        idf = self.compute_idf(texts)
        tf = self.compute_tf(texts)
        X = tf * idf
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        norms[norms == 0] = 1
        X = X / norms
        return X

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.random.rand(n_features) * 0.01
        self.bias = np.random.rand() * 0.01

        for _ in range(self.n_iters):
            for i in range(n_samples):
                margin = y[i] * (np.dot(X[i], self.weights) + self.bias)
                if margin < 1:
                    self.weights -= self.learning_rate * (2 * self.lambda_param * self.weights - np.dot(X[i], y[i]))
                    self.bias += self.learning_rate * y[i]
                else:
                    self.weights -= self.learning_rate * 2 * self.lambda_param * self.weights

    def predict(self, X):
        return np.sign(np.dot(X, self.weights) + self.bias)
