import nltk
import numpy as np
import pandas as pd
import pandas.core.series
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.util import ngrams
from collections import Counter
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(word):
    return lemmatizer.lemmatize(word)


def trigram_token(text):
    tokens = word_tokenize(text.lower())
    filtered_words = [
        preprocess(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    trigrams = list(ngrams(filtered_words, 3))
    return trigrams


def build_vocab(texts, top_n=5000):
    trigram_counts = Counter()
    for text in texts:
        if not isinstance(text, pandas.core.series.Series):
            text = eval(text)
        trigram_counts.update(text)

    vocab = {k: v for k, v in trigram_counts.most_common(top_n)}
    return vocab


def vectorize_text(text, vocab):
    vector = np.zeros(len(vocab))
    if not isinstance(text, pandas.core.series.Series):
        text = eval(text)
    index_map = {trigram: idx for idx, trigram in enumerate(vocab.keys())}
    for trigram in text:
        if trigram in index_map:
            vector[index_map[trigram]] += 1
    return vector


def fit(X, y, n_iters=1000, learning_rate=0.01, lambda_param=0.01):
    n_samples, n_features = X.shape
    weights = np.random.rand(n_features) * 0.01
    bias = np.random.rand() * 0.01

    for _ in range(n_iters):
        for i in range(n_samples):
            margin = y[i] * (np.dot(X[i], weights) + bias)
            if margin < 1:
                weights -= learning_rate * (2 * lambda_param * weights - np.dot(X[i], y[i]))
                bias += learning_rate * y[i]
            else:
                weights -= learning_rate * 2 * lambda_param * weights
    return weights, bias


def predict(X, weights, bias):
    approx = np.dot(X, weights) + bias
    return np.sign(approx)


learning_rate = 0.01
lambda_param = 0.0002
n_iters = 2000

fake = 'fake.csv'
real = 'true.csv'
processed_data_path = 'processed_data.csv'
csvs = [fake, real]


def calculate_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


if __name__ == '__main__':
    if os.path.exists(processed_data_path):
        df = pd.read_csv(processed_data_path)
    else:
        filtered_words_list = []
        for i, csv in enumerate(csvs):
            df = pd.read_csv(csv)
            df['label'] = -1 if i == 0 else 1
            df['processed_text'] = df['text'].dropna().apply(trigram_token)
            print(type(df['processed_text']))
            filtered_words_list.append(df)
        df = pd.concat(filtered_words_list)
        df.to_csv(processed_data_path, index=False)

    df = df.sample(frac=1, random_state=44).reset_index(drop=True)
    vocab = build_vocab(df['processed_text'], 5000)
    X = np.array([vectorize_text(text, vocab) for text in df['processed_text']])
    y = df['label'].values

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    train_size = int(0.7 * X.shape[0])
    train_indices = indices[:train_size]
    test_indices = indices[train_size:]

    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]

    print(f"Total Size: {X.shape}")
    print(f"Training data size: {X_train.shape}, Testing data size: {X_test.shape}")

    weights, bias = fit(X_train, y_train, learning_rate=learning_rate, lambda_param=lambda_param, n_iters=n_iters)
    pred = predict(X_test, weights, bias)
    accuracy = np.mean(pred == y_test)
    print(f"Accuracy: {accuracy}")
    precision, recall, f1_score = calculate_metrics(y_test, pred)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
