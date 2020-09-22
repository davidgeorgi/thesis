import argparse
parser = argparse.ArgumentParser(description='Evaluate Text-Aware Process Prediction')
parser.add_argument('log',
                    help='an event log in XES format (.xes)')
parser.add_argument('-a', '--attributes', nargs='+', required=False,
                    help='list of considered numerical or categorical attributes besides activity and timestamp')
parser.add_argument('-t', '--text', nargs=1, required=True,
                    help='attribute name with textual data')
parser.add_argument('-l', '--language', nargs=1, required=False,
                    help='language of the text in the log')

args = parser.parse_args()

print("Prepare...")
from pm4py.objects.log.importer.xes import importer as xes_importer
from tapp.tapp_model import TappModel
from tapp.log_encoder import LogEncoder
from tapp.text_encoder import BoWTextEncoder
from tapp.text_encoder import BoNGTextEncoder
from tapp.text_encoder import PVTextEncoder
from tapp.text_encoder import LDATextEncoder
from nltk.tokenize import word_tokenize
from pm4py.algo.filtering.log.variants import variants_filter
import numpy as np
import pandas as pd
import nltk

# Download text preprocessing resources from nltk
try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet")
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt")
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

print("Done.")

# Load event data
print("Load event log...")
path = args.log
variant = xes_importer.Variants.ITERPARSE
parameters = {variant.value.Parameters.TIMESTAMP_SORT: True, variant.value.Parameters.REVERSE_SORT: False}
log = xes_importer.apply(path, variant=variant, parameters=parameters)
print("Done.")

# Analyse log
print("Create log statistics...")
language = "english" if args.language is None else args.language[0]
text_attribute = args.text[0]
traces = len(log)
events = sum(len(case) for case in log)
durations = [(case[-1]["time:timestamp"].timestamp() - case[0]["time:timestamp"].timestamp())/86400 for case in log]
docs = [event[text_attribute] for case in log for event in case if text_attribute in event]
words = [word for doc in docs for word in word_tokenize(doc, language=language)]
docs_filtered = BoWTextEncoder().preprocess_docs(docs, as_list=False)
words_filtered = [word for doc in docs_filtered for word in word_tokenize(doc, language=language)]

log_info = pd.DataFrame([[path,
                          traces,
                          len(variants_filter.get_variants(log)),
                          events,
                          events/traces,
                          np.median(durations),
                          np.mean(durations),
                          len(list(dict.fromkeys([event["concept:name"] for case in log for event in case])) if log else []),
                          len(words),
                          len(words_filtered),
                          len(set(words)),
                          len(set(words_filtered))]],
                        columns=["log", "cases", "trace variants", "events", "events per trace", "median case duration", "mean case duration", "activities", "words pre filtering", "words post filtering", "vocabulary pre filtering", "vocabulary post filtering"])
log_info.to_csv("log_info.csv", index=False, sep=";")
print("Done.")

# Split data in train and test log
split = len(log) // 3 * 2
train_log = log[:split]
test_log = log[split:]

# Configure and build model variants
language = "english"
text_models = [
    None,
    BoWTextEncoder(encoding_length=50, language=language),
    BoWTextEncoder(encoding_length=100, language=language),
    BoWTextEncoder(encoding_length=250, language=language),
    BoNGTextEncoder(n=2, encoding_length=50, language=language),
    BoNGTextEncoder(n=2, encoding_length=100, language=language),
    BoNGTextEncoder(n=2, encoding_length=250, language=language),
    PVTextEncoder(encoding_length=5, language=language),
    PVTextEncoder(encoding_length=10, language=language),
    PVTextEncoder(encoding_length=25, language=language),
    LDATextEncoder(encoding_length=5, language=language),
    LDATextEncoder(encoding_length=10, language=language),
    LDATextEncoder(encoding_length=25, language=language),
]

shared_layers = [1]
special_layers = [1]
neurons = [100]
data_attributes_list = [[]] if args.attributes is None else [args.attributes]

print("Evaluate prediction models...")
print("This might take a while...")
for text_model in text_models:
    for shared_layer in shared_layers:
        for special_layer in special_layers:
            for neuron in neurons:
                for data_attributes in data_attributes_list:
                    if shared_layer + special_layer == 0:
                        pass
                    else:
                        log_encoder = LogEncoder(text_encoder=text_model, advanced_time_attributes=True)
                        model = TappModel(log_encoder=log_encoder, num_shared_layer=shared_layer, num_specialized_layer=special_layer, neurons_per_layer=neuron, dropout=0.2, learning_rate=0.001)
                        model.fit(train_log, data_attributes=data_attributes, text_attribute=text_attribute, epochs=100)
                        model.evaluate(test_log, "./results.csv", num_prefixes=8)
print("Done. Evaluation completed.")
