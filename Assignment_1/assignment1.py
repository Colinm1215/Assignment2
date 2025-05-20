import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from collections import Counter

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt_tab')

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))


def preprocess(word):
    return lemmatizer.lemmatize(word)


fake = 'fake.csv'
real = 'true.csv'
csvs = [fake, real]
text_column = 'text'

filtered_words_list = []
for csv in csvs:
    df = pd.read_csv(csv)
    all_text = ' '.join(df[text_column].dropna())
    tokens = word_tokenize(all_text.lower())
    filtered_words = [
        preprocess(word)
        for word in tokens
        if word.isalpha() and word not in stop_words
    ]
    filtered_words_list.append(filtered_words)

word_counts = Counter(filtered_words_list[0])
top_100_words = word_counts.most_common(100)
print("Top 100 most commonly used words in fake news:")
for word, count in top_100_words:
    print(f"{word}: {count}")
word_counts = Counter(filtered_words_list[1])
top_100_words = word_counts.most_common(100)
print("Top 100 most commonly used words in real news:")
for word, count in top_100_words:
    print(f"{word}: {count}")
word_counts = Counter(filtered_words_list[0] + filtered_words_list[1])
top_100_words = word_counts.most_common(100)
print("Top 100 most commonly used words in all news:")
for word, count in top_100_words:
    print(f"{word}: {count}")
