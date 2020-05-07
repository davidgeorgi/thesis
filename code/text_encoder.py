import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer


class BoNGTextEncoder:

    def __init__(self, n, encoding_length=100, language="english"):
        self.encoding_length = encoding_length
        self.language = language
        self.n = n
        self.vectorizer = None

    def fit(self, docs):
        self.vectorizer = TfidfVectorizer(ngram_range=(self.n, self.n), max_features=self.encoding_length, analyzer='word', norm="l2")
        return self.vectorizer.fit(preprocess_docs(docs, language=self.language))

    def transform(self, docs):
        self.vectorizer.transform(preprocess_docs(docs, language=self.language))


def preprocess_docs(docs, language="english"):
    nltk.download('wordnet')
    nltk.download('punkt')
    stop_words = set(stopwords.words(language))
    lemmatizer = WordNetLemmatizer()
    docs_preprocessed = []
    for document in docs:
        document = document.lower()
        words = word_tokenize(document)
        docs_preprocessed.append(" ".join([lemmatizer.lemmatize(word) for word in words if not word in stop_words and word.isalpha()]))
    return docs_preprocessed
