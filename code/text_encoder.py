import multiprocessing
import nltk
import numpy as np
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument


class BoNGTextEncoder:

    def __init__(self, n, encoding_length=50, language="english"):
        self.encoding_length = encoding_length
        self.language = language
        self.n = n
        self.vectorizer = None

    def fit(self, docs):
        self.vectorizer = TfidfVectorizer(ngram_range=(self.n, self.n), max_features=self.encoding_length, analyzer='word', norm="l2")
        return self.vectorizer.fit(preprocess_docs(docs, language=self.language))

    def transform(self, docs):
        self.vectorizer.transform(preprocess_docs(docs, language=self.language))


class PVTextEncoder:

    def __init__(self, encoding_length=50, language="english", epochs=10, min_count=2):
        self.encoding_length = encoding_length
        self.language = language
        self.epochs = epochs
        self.min_count = min_count
        self.workers = multiprocessing.cpu_count()
        self.model = None

    def fit(self, docs):
        docs = preprocess_docs(docs, language=self.language)

        tagged_docs = [TaggedDocument(words=word_tokenize(doc), tags=[i]) for i, doc in enumerate(docs)]

        self.model = Doc2Vec(dm=0, vector_size=self.encoding_length, negative=5, hs=0, min_count=self.min_count, sample=0, workers=self.workers)
        self.model.build_vocab(tagged_docs)

        for epoch in self.epochs:
            self.model.train(utils.shuffle(tagged_docs), total_examples=len(tagged_docs), epochs=1)
            self.model.alpha -= 0.002
            self.model.min_alpha = self.model.alpha

    def transform(self, docs):
        docs = preprocess_docs(docs, language=self.language)
        return np.array([self.model.infer_vector(word_tokenize(doc)) for doc in docs])


def preprocess_docs(docs, language="english"):
    nltk.download('wordnet')
    nltk.download('punkt')
    stop_words = set(stopwords.words(language))
    lemmatizer = WordNetLemmatizer()
    docs_preprocessed = []
    for doc in docs:
        doc = doc.lower()
        words = word_tokenize(doc)
        docs_preprocessed.append(" ".join([lemmatizer.lemmatize(word) for word in words if not word in stop_words and word.isalpha()]))
    return docs_preprocessed
