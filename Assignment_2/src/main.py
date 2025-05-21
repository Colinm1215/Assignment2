import kagglehub
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import os
import nltk
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from classifiers.meanword2vec import MeanWord2Vec
from classifiers.maxword2vec import MaxWord2Vec
from classifiers.tfidfword2vec import TFIDFWord2Vec
from classifiers.lorabert import LoRABERT
from classifiers.bert import BERT
from classifiers.tfidf import TFIDF
from classifiers.bertfinetune import FineTunedBERT
import torch
import numpy as np

def load_data():
    path = kagglehub.dataset_download("snap/amazon-fine-food-reviews")
    data_file = os.path.join(path, "Reviews.csv")
    df = pd.read_csv(data_file)
    return df

def sample_and_label(df, frac=0.2):
    df = df[["Score", "Text"]].dropna()
    df = df[df["Score"] != 3]
    df["label"] = df["Score"].apply(lambda x: 1 if x > 3 else 0)
    df = df.sample(frac=frac, random_state=42)
    return df

def balance_classes(df):
    pos = df[df.label == 1]
    neg = df[df.label == 0]
    if len(pos) > len(neg):
        pos_down = resample(pos, replace=False, n_samples=len(neg), random_state=42)
        df_bal = pd.concat([pos_down, neg])
    else:
        neg_down = resample(neg, replace=False, n_samples=len(pos), random_state=42)
        df_bal = pd.concat([neg_down, pos])
    return df_bal.sample(frac=1, random_state=42)

def split_data(df):
    return train_test_split(df["Text"].tolist(), df["label"].tolist(), test_size=0.2, random_state=42)

def evaluate(y_true, y_pred, name):
    print(f"\n=== Results: {name} ===")
    print(f"Accuracy : {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall   : {recall_score(y_true, y_pred):.4f}")
    print(f"F1 Score : {f1_score(y_true, y_pred):.4f}")

if __name__ == "__main__":
    df = load_data()
    df = sample_and_label(df, frac=0.2)
    df_bal = balance_classes(df)
    X_train, X_test, y_train, y_test = split_data(df_bal)

    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
    print(f"Label distribution:\n{pd.Series(y_train + y_test).value_counts(normalize=True)}")
    print(f"Sample text: {X_train[0]}")
    print(f"Sample label: {y_train[0]}")

    device = 0 if torch.cuda.is_available() else -1

    models = {
        "TFIDF-1grams": TFIDF(ngram_range=(1,1)),
        "TFIDF-(1,2)grams": TFIDF(ngram_range=(1,2)),
        "TFIDF-(1,2,3)grams": TFIDF(ngram_range=(1,3)),
        "Word2Vec-MeanPooling": MeanWord2Vec(),
        "Word2Vec-MaxPooling": MaxWord2Vec(),
        "Word2Vec-TFIDF": TFIDFWord2Vec(),
        "BERT-NoFineTune": BERT(device=device),
        "BERT-FineTune": FineTunedBERT(device=device),
        "BERT-LoRA": LoRABERT(device=device),
    }

    preds_dict = {}

    for name, model in models.items():
        print(f"\n=== {name} ===")

        if name == "BERT-NoFineTune":
            print(f"No preprocessing or training needed for {name}.")

            X_train_vec = X_train
            X_test_vec = X_test

        elif name == "BERT-FineTune" or name == "BERT-LoRA":
            print(f"No preprocessing needed for {name}.")

            X_train_vec = X_train
            X_test_vec = X_test

            print(f"Fine-tuning BERT model for {name}...")
            model.fit(X_train_vec, y_train)
        elif name.__contains__("TFIDF") and not name.__contains__("Word2Vec"):
            print(f"Preprocessing data for {name}...")
            y_train_bin = np.where(np.array(y_train) == 1, 1, -1)
            y_test_bin = np.where(np.array(y_test) == 1, 1, -1)

            X_train_clean = model.preprocess(X_train)
            X_test_clean = model.preprocess(X_test)

            X_train_vec = model.fit_transform(X_train_clean)
            X_test_vec = model.transform(X_test_clean)

            print(f"Training {name}...")

            model.fit(X_train_vec, y_train_bin)

        else:
            print(f"Preprocessing data for {name}...")

            X_train_clean = model.preprocess(X_train)
            X_test_clean = model.preprocess(X_test)
            X_train_vec = model.build_features(X_train_clean)
            X_test_vec = model.build_features(X_test_clean)

            print(f"Training {name}...")

            model.fit(X_train_vec, y_train)

        print(f"Predicting with {name}...")

        preds = model.predict(X_test_vec)

        if set(np.unique(preds)) == {-1, 1}:
            preds = np.where(preds == 1, 1, 0)

        preds_dict[name] = preds

        print(f"=== Finished {name} ===")

    for name, preds in preds_dict.items():
        evaluate(y_test, preds, name)