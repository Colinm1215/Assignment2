import math
import nltk
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter
import os

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(word):
    return lemmatizer.lemmatize(word)

def token(text):
    tokens = word_tokenize(text.lower())
    filtered_words = [
        preprocess(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    return filtered_words

def compute_tf(texts, vocab):
    n_docs = len(texts)
    vocab_list = list(vocab.keys())
    tf_matrix = np.zeros((n_docs, len(vocab_list)))

    for i, text in enumerate(texts):
        tf_c = Counter(text)
        n = len(text)
        for (word, count) in tf_c.items():
            if word in vocab:
                tf_matrix[i, vocab_list.index(word)] = count / n

    return tf_matrix


def compute_idf(texts, vocab):
    n_docs = len(texts)
    vocab_list = list(vocab.keys())
    doc_freq = np.zeros(len(vocab_list))

    for text in texts:
        unique_words = set(text)
        for word in unique_words:
            if word in vocab:
                doc_freq[vocab_list.index(word)] += 1

    idf_vector = np.log((n_docs + 1) / (doc_freq + 1)) + 1
    return idf_vector


def build_vocab(texts, top_n=10):
    word_counter = Counter()
    for text in texts:
        try:
            word_counter.update(text)
        except TypeError as e:
            print(f"TypeError: {e} : {text}")
    vocab = {}
    for word, count in word_counter.most_common(top_n):
        vocab.update({word: count})
    return vocab


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


def calculate_metrics(y_true, y_pred):
    TP = np.sum((y_true == 1) & (y_pred == 1))
    TN = np.sum((y_true == -1) & (y_pred == -1))
    FP = np.sum((y_true == -1) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == -1))

    precision = TP / (TP + FP) if (TP + FP) > 0 else 0
    recall = TP / (TP + FN) if (TP + FN) > 0 else 0
    f1_score = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

    return precision, recall, f1_score


def compute_confusion_matrix(y_true, y_pred):
    classes = np.unique(np.concatenate((y_true, y_pred)))
    num_classes = len(classes)

    label_to_index = {label: i for i, label in enumerate(classes)}

    cm = np.zeros((num_classes, num_classes), dtype=int)

    for true, pred in zip(y_true, y_pred):
        if not np.isnan(pred):
            cm[label_to_index[true], label_to_index[pred]] += 1

    return cm, classes


def plot_confusion_matrix(cm, classes):
    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix TFIDF')
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    thresh = cm.max() / 2
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    plt.show()


learning_rate = 0.01
lambda_param = 0.0002
n_iters = 2000

fake = 'fake.csv'
real = 'true.csv'
processed_data_path = 'processed_data_2.csv'
csvs = [fake, real]

if __name__ == '__main__':
    if os.path.exists(processed_data_path):
        df = pd.read_csv(processed_data_path)
        df = df.dropna(subset=['processed_text'])
        texts = df['processed_text'].apply(str.split)
        vocab = build_vocab(texts, top_n=100)
        idf = compute_idf(texts, vocab)
        tf = compute_tf(texts, vocab)
        X = tf * idf
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
        y = df['label'].values
    else:
        filtered_words_list = []
        for i, csv in enumerate(csvs):
            df = pd.read_csv(csv)
            df['label'] = -1 if i == 0 else 1
            df['processed_text'] = df["text"].dropna().apply(token)
            filtered_words_list.append(df)
        df = pd.concat(filtered_words_list)
        df['interleave_index'] = (df.index // 2)
        df = df.sort_values(by='interleave_index')
        df = df.dropna(subset=['processed_text'])
        df.to_csv(processed_data_path, index=False)
        texts = df['processed_text'].apply(str.split)
        vocab = build_vocab(texts, top_n=100)
        idf = compute_idf(texts, vocab)
        tf = compute_tf(texts, vocab)
        X = tf * idf
        X = X / np.linalg.norm(X, axis=1, keepdims=True)
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
    print(X_train)
    print(y_train)

    weights, bias = fit(X_train, y_train)

    pred = predict(X_test, weights, bias)
    print(f"Accuracy: {np.mean(pred == y_test)}")
    precision, recall, f1_score = calculate_metrics(y_test, pred)
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")
    cm, classes = compute_confusion_matrix(y_test, pred)
    plot_confusion_matrix(cm, classes)
