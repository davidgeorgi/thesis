from __future__ import absolute_import, division, print_function, unicode_literals
from pm4py.objects.log.importer.xes import factory as xes_import_factory
import datetime
import sys
import collections
import matplotlib.pyplot as plt
import numpy as np
from numpy import argmax
import datetime
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
import tensorflow_addons as tfa
from tensorflow_addons.metrics import F1Score
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
from nltk.stem import SnowballStemmer
from nltk.corpus import stopwords


def load_log(path):
    parameters = {"timestamp_sort": True}
    log = xes_import_factory.apply(path, variant="nonstandard", parameters=parameters)
    return log

def get_event_attributes(log):
    return list(log[0][0].keys()) if log else []

def get_case_attributes(log):
    return list(log[0].attributes.keys()) if log else []

def get_max_case_length(log):
    return max([len(case) for case in log])

def get_eventlist(log):
    return [dict(event) for case in enumerate(log) for event in enumerate(case)] if log else []

def get_activities(log):
    return list(dict.fromkeys([dict(event)['concept:name'] for case in log for event in enumerate(case)] if log else []))


def docs_to_vecs_ngram(docs, n=2, language="english"):
    stemmer = SnowballStemmer(language, ignore_stopwords=True)
    tokenizer = TfidfVectorizer().build_tokenizer()

    def toknizer_with_stemming(doc):
        return [stemmer.stem(word) for word in tokenizer(doc)]

    vectorizer = TfidfVectorizer(tokenizer=toknizer_with_stemming, lowercase=True, stop_words=stopwords.words(language),
                                 ngram_range=(n, n), max_features=None, analyzer='word', norm="l2")
    vectors = vectorizer.fit_transform(docs)
    return vectorizer, vectors.toarray()


def vectorize_log(log):

    activities = get_activities(log)

    # number of all prefixes
    case_dim = sum([len(case) for case_index, case in enumerate(log)])

    # length of the longest case
    event_dim = get_max_case_length(log)

    # number of activites + 2 features for time
    data_dim = len(activities) + 2

    x = np.zeros((case_dim, event_dim, data_dim))

    y_next_act = np.zeros((case_dim, len(activities) + 1))
    y_final_act = np.zeros((case_dim, len(activities) + 1))
    y_next_time = np.zeros(case_dim)
    y_final_time = np.zeros(case_dim)

    time_since_start_normalization = max(
        [case[-1]["time:timestamp"].timestamp() - case[0]["time:timestamp"].timestamp() for case_index, case in
         enumerate(log)])

    times_since_last_event = []
    for case_index, case in enumerate(log):
        last_event_time = 0
        for event_index, event in enumerate(case):
            current_event_time = event["time:timestamp"].timestamp()
            times_since_last_event.append(current_event_time - last_event_time)
            last_event_time = current_event_time
    time_since_last_event_normalization = max(times_since_last_event)

    trace_dim_index = 0
    for case_index, case in enumerate(log):
        case_start_time = log[case_index][0]["time:timestamp"].timestamp()
        for prefix_length in range(1, len(case) + 1):
            last_event_time = 0
            for event_index, event in enumerate(case):

                if event_index <= prefix_length - 1:
                    x[trace_dim_index][event_index][activities.index(event["concept:name"])] = 1
                    x[trace_dim_index][event_index][len(activities)] = (event[
                                                                            "time:timestamp"].timestamp() - case_start_time) / time_since_start_normalization
                    x[trace_dim_index][event_index][len(activities) + 1] = (event[
                                                                                "time:timestamp"].timestamp() - last_event_time) / time_since_last_event_normalization

                last_event_time = event["time:timestamp"].timestamp()

            if prefix_length == len(case):
                # set <Process end> as next activity
                y_next_act[trace_dim_index][len(activities)] = 1
                y_next_time[trace_dim_index] = (case[prefix_length - 1][
                                                    "time:timestamp"].timestamp() - case_start_time) / time_since_start_normalization
            else:
                # set next activity target
                y_next_act[trace_dim_index][activities.index(case[prefix_length]["concept:name"])] = 1
                y_next_time[trace_dim_index] = (case[prefix_length][
                                                    "time:timestamp"].timestamp() - case_start_time) / time_since_start_normalization

            y_final_act[trace_dim_index][activities.index(case[-1]["concept:name"])] = 1
            y_final_time[trace_dim_index] = (case[-1][
                                                 "time:timestamp"].timestamp() - case_start_time) / time_since_start_normalization
            trace_dim_index += 1
    return x, y_next_act, y_final_act, y_next_time, y_final_time


def build_model(timesteps, data_dim, output_dim):
    inputs = tf.keras.Input(shape=(timesteps, data_dim), name='inputs')

    shared_lstm_layer = layers.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True,
                                    dropout=0.2)(inputs)

    shared_lstm_normalization_layer = layers.BatchNormalization()(shared_lstm_layer)

    next_activity_lstm_layer = layers.LSTM(100, implementation=2, kernel_initializer='glorot_uniform',
                                           return_sequences=False, dropout=0.2)(shared_lstm_normalization_layer)
    final_activity_lstm_layer = layers.LSTM(100, implementation=2, kernel_initializer='glorot_uniform',
                                            return_sequences=False, dropout=0.2)(shared_lstm_normalization_layer)
    next_timestamp_lstm_layer = layers.LSTM(100, implementation=2, kernel_initializer='glorot_uniform',
                                            return_sequences=False, dropout=0.2)(shared_lstm_normalization_layer)
    final_timestamp_lstm_layer = layers.LSTM(100, implementation=2, kernel_initializer='glorot_uniform',
                                             return_sequences=False, dropout=0.2)(shared_lstm_normalization_layer)

    next_activity_normalization_layer = layers.BatchNormalization()(next_activity_lstm_layer)
    final_activity_normalization_layer = layers.BatchNormalization()(final_activity_lstm_layer)
    next_timestamp_normalization_layer = layers.BatchNormalization()(next_timestamp_lstm_layer)
    final_timestamp_normalization_layer = layers.BatchNormalization()(final_timestamp_lstm_layer)

    next_activity_output = layers.Dense(output_dim, activation='softmax', kernel_initializer='glorot_uniform',
                                        name='next_activity_output')(next_activity_normalization_layer)
    final_activity_output = layers.Dense(output_dim, activation='softmax', kernel_initializer='glorot_uniform',
                                         name='final_activity_output')(final_activity_normalization_layer)
    next_timestamp_ouput = layers.Dense(1, activation='relu', kernel_initializer='glorot_uniform',
                                        name='next_timestamp_ouput')(next_timestamp_normalization_layer)
    final_timestamp_ouput = layers.Dense(1, activation='relu', kernel_initializer='glorot_uniform',
                                         name='final_timestamp_ouput')(final_timestamp_normalization_layer)

    model = tf.keras.Model(inputs=[inputs], outputs=[next_activity_output, final_activity_output, next_timestamp_ouput,
                                                     final_timestamp_ouput])

    optimizer_param = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07,
                                                name='Nadam')
    metric_param = {'next_activity_output': metrics.CategoricalAccuracy(),
                    'final_activity_output': metrics.CategoricalAccuracy(),
                    'next_timestamp_ouput': metrics.MeanAbsoluteError(),
                    'final_timestamp_ouput': metrics.MeanAbsoluteError()}
    loss_param = {'next_activity_output': 'categorical_crossentropy',
                  'final_activity_output': 'categorical_crossentropy', 'next_timestamp_ouput': 'mae',
                  'final_timestamp_ouput': 'mae'}

    model.compile(loss=loss_param, metrics=metric_param, optimizer=optimizer_param)

    return model
