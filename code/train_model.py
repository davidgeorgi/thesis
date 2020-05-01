from pm4py.objects.log.importer.xes import factory as xes_import_factory
from pm4py.objects.log.util.log import get_event_labels

import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import numpy as np

from sklearn.preprocessing import StandardScaler

import matplotlib.pyplot as plt


# Helper functions
def load_log(path):
    parameters = {"timestamp_sort": True}
    log = xes_import_factory.apply(path, variant="nonstandard", parameters=parameters)
    return log

def is_numerical(value):
  try:
    float(value)
    return True
  except ValueError:
    return False

def is_numerical_attribute(log, attribute):
    return is_numerical(log[0][0][attribute])

def get_max_case_length(log):
    return max([len(case) for case in log]) if log else 0


def vectorize_log(log, additional_attributes, text_attribute):

    # Get activities
    activities = get_event_labels(log, "concept:name")

    # Get list of categorical attributes
    categorical_attributes = list(filter(lambda attribute: not is_numerical_attribute(log, attribute), additional_attributes))

    # Get values of categoical attributes
    categorical_attributes_values = [get_event_labels(log, attribute) for attribute in categorical_attributes]

    # Get numerical attributes
    numerical_attributes = list(filter(lambda attribute: is_numerical_attribute(log, attribute), additional_attributes))

    # Case dimension: Number of all prefixes of all traces
    case_dim = sum([len(case) for case_index, case in enumerate(log)])

    # Event dimenstion: Length of the longest trace
    event_dim = get_max_case_length(log)

    #   Feature dimension: Encoding size of an event
    # = Number of activites (1-hot)
    # + 6 features for time
    # + Size of encodings of additional categorical attributes (1-hot)
    # + Number of additional numerical attributes
    # + Size of text encoding
    feature_dim = len(activities) + 6 + sum([len(values) for values in categorical_attributes_values]) + len(numerical_attributes) + 0

    # Prepare input and output vectors/matrices
    X = np.zeros((case_dim, event_dim, feature_dim))
    y_next_act = np.zeros((case_dim, len(activities) + 1))
    y_final_act = np.zeros((case_dim, len(activities)))
    y_next_time = np.zeros(case_dim)
    y_final_time = np.zeros(case_dim)

    # Scaling divisors for time related features to archieve values between 0 and 1
    cycle_time_max = np.max([case[-1]["time:timestamp"].timestamp() - case[0]["time:timestamp"].timestamp() for case in log])
    time_between_events_max = np.max([event["time:timestamp"].timestamp() - case[event_index - 1]["time:timestamp"].timestamp() for case in log for event_index, event in enumerate(case) if event_index > 0])
    process_end_time = np.max([event["time:timestamp"].timestamp() for case in log for event in case])
    time_scaling_divisor = [cycle_time_max, time_between_events_max, 86400, 604800, 31536000, process_end_time]

    # Encode traces and prefix traces
    trace_dim_index = 0
    process_start_time = log[0][0]["time:timestamp"].timestamp()
    for case_index, case in enumerate(log):
        case_start_time = log[case_index][0]["time:timestamp"].timestamp()
        for prefix_length in range(1, len(case) + 1):

            # Encode the (prefix-)trace
            previous_event_time = case_start_time
            for event_index, event in enumerate(case):

                if event_index <= prefix_length - 1:
                    # Set 1-hot activity
                    X[trace_dim_index][event_index][activities.index(event["concept:name"])] = 1
                    offset = len(activities)

                    # Set time attributes
                    event_time = event["time:timestamp"]
                    # Seconds since case start
                    X[trace_dim_index][event_index][offset] = (event_time.timestamp() - case_start_time)/time_scaling_divisor[0]
                    # Seconds since previous event
                    X[trace_dim_index][event_index][offset+1] = (event_time.timestamp() - previous_event_time)/time_scaling_divisor[1]
                    # Seconds since midnight
                    X[trace_dim_index][event_index][offset+2] = (event_time.timestamp() - event_time.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())/time_scaling_divisor[2]
                    # Seconds since last Monday
                    X[trace_dim_index][event_index][offset+3] = (event_time.weekday() * 86400 + event_time.hour * 3600 + event_time.second)/time_scaling_divisor[3]
                    # Seconds since last Januar 1
                    X[trace_dim_index][event_index][offset+4] = (event_time.timestamp() - event_time.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0).timestamp())/time_scaling_divisor[4]
                    # Seconds since process start
                    X[trace_dim_index][event_index][offset+5] = (event_time.timestamp() - process_start_time)/time_scaling_divisor[5]

                    previous_event_time = event_time.timestamp()
                    offset += 6

                    # Set categorical attributes
                    for attribute_index, attribute in enumerate(categorical_attributes):
                        X[trace_dim_index][event_index][offset + categorical_attributes_values[attribute_index].index(event[attribute])] = 1
                        offset += len(categorical_attributes_values[attribute_index])

                    # Set numerical attributes
                    for attribute_index, attribute in enumerate(numerical_attributes):
                        X[trace_dim_index][event_index][offset] = float(event[attribute])
                        offset += 1

            # Set activity and time (since case start) of next event as target
            if prefix_length == len(case):
                # Case 1: Set <Process end> as next activity target
                y_next_act[trace_dim_index][len(activities)] = 1
                y_next_time[trace_dim_index] = (case[-1]["time:timestamp"].timestamp() - case_start_time)/time_between_events_max
            else:
                # Case 2: Set next activity as target
                y_next_act[trace_dim_index][activities.index(case[prefix_length]["concept:name"])] = 1
                y_next_time[trace_dim_index] = (case[prefix_length]["time:timestamp"].timestamp() - case_start_time)/time_between_events_max
            # Set final activity and case cycle time as target
            y_final_act[trace_dim_index][activities.index(case[-1]["concept:name"])] = 1
            y_final_time[trace_dim_index] = (case[-1]["time:timestamp"].timestamp() - case_start_time)/cycle_time_max

            # Increase index for next (prefix-)trace
            trace_dim_index += 1

    return X, y_next_act, y_final_act, y_next_time, y_final_time


def build_model(timesteps, feature_dim, number_of_activities):
    inputs = tf.keras.Input(shape=(timesteps, feature_dim),  name='inputs')

    shared_lstm_layer = layers.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(inputs)

    shared_lstm_normalization_layer = layers.BatchNormalization()(shared_lstm_layer)

    next_activity_lstm_layer = layers.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(shared_lstm_normalization_layer)
    final_activity_lstm_layer = layers.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(shared_lstm_normalization_layer)
    next_timestamp_lstm_layer = layers.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(shared_lstm_normalization_layer)
    final_timestamp_lstm_layer = layers.LSTM(100, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(shared_lstm_normalization_layer)

    next_activity_normalization_layer = layers.BatchNormalization()(next_activity_lstm_layer)
    final_activity_normalization_layer = layers.BatchNormalization()(final_activity_lstm_layer)
    next_timestamp_normalization_layer = layers.BatchNormalization()(next_timestamp_lstm_layer)
    final_timestamp_normalization_layer = layers.BatchNormalization()(final_timestamp_lstm_layer)

    next_activity_output = layers.Dense(number_of_activities + 1, activation='softmax', kernel_initializer='glorot_uniform', name='next_activity_output')(next_activity_normalization_layer)
    final_activity_output = layers.Dense(number_of_activities, activation='softmax', kernel_initializer='glorot_uniform', name='final_activity_output')(final_activity_normalization_layer)
    next_timestamp_ouput = layers.Dense(1, kernel_initializer='glorot_uniform', name='next_timestamp_output')(next_timestamp_normalization_layer)
    final_timestamp_ouput = layers.Dense(1, kernel_initializer='glorot_uniform', name='final_timestamp_output')(final_timestamp_normalization_layer)

    # Build and configure model
    model = tf.keras.Model(inputs=[inputs], outputs=[next_activity_output, final_activity_output, next_timestamp_ouput, final_timestamp_ouput])

    optimizer_param = tf.keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')
    metric_param = {'next_activity_output':metrics.CategoricalAccuracy(), 'final_activity_output':metrics.CategoricalAccuracy(), 'next_timestamp_output':metrics.MeanAbsoluteError(), 'final_timestamp_output': metrics.MeanAbsoluteError()}
    loss_param = {'next_activity_output':'categorical_crossentropy', 'final_activity_output':'categorical_crossentropy', 'next_timestamp_output':'mae', 'final_timestamp_output':'mae'}

    model.compile(loss=loss_param, metrics=metric_param, optimizer=optimizer_param)

    return model



path_to_data = "../data/logs/"
file_name = "hospital_billing_filtered.xes"
log = load_log(path_to_data + file_name)
X, y_next_act, y_final_act, y_next_time, y_final_time = vectorize_log(log, [], '')
event_dim = X.shape[1]
feature_dim = X.shape[2]
number_of_activities = len(get_event_labels(log, "concept:name"))
model = build_model(event_dim, feature_dim, number_of_activities)
history = model.fit(X, {'next_activity_output': y_next_act, 'final_activity_output': y_final_act, 'next_timestamp_output': y_next_time, 'final_timestamp_output': y_final_time}, epochs=100, batch_size=None, validation_split=0.2)
