from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    tokens = word_tokenize(text.lower())
    filtered = [
        lemmatizer.lemmatize(w)
        for w in tokens if w.isalpha() and w not in stop_words
    ]
    return ' '.join(filtered)

def clean_texts(texts):
    return [clean_text(text) for text in texts]