import numpy as np
from abc import ABC, abstractmethod


class LogEncoder(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, docs):
        pass

    @abstractmethod
    def transform(self, docs):
        pass


class LSTMLogEncoder(LogEncoder):
    
    def __init__(self, text_encoder=None):
        self.text_encoder = text_encoder
        self.activities = []
        self.data_attributes = []
        self.text_attribute = None
        self.categorical_attributes = []
        self.categorical_attributes_values = []
        self.numerical_attributes = []
        self.event_dim = 0
        self.feature_dim = 0
        self.time_scaling_divisor = [1, 1, 1, 1, 1, 1]
        self.process_start_time = 0
        super().__init__()

    def fit(self, log, activities=None, data_attributes=None, text_attribute=None):
        # Fit encoder to log
        self.activities = activities
        self.data_attributes = data_attributes
        self.text_attribute = text_attribute
        self.categorical_attributes = list(filter(lambda attribute: not _is_numerical_attribute(log, attribute), self.data_attributes))
        self.categorical_attributes_values = [_get_event_labels(log, attribute) for attribute in self.categorical_attributes]
        self.numerical_attributes = list(filter(lambda attribute: _is_numerical_attribute(log, attribute), self.data_attributes))
        self.process_start_time = log[0][0]["time:timestamp"].timestamp()

        # Scaling divisors for time related features to achieve values between 0 and 1
        cycle_time_max = np.max(
            [case[-1]["time:timestamp"].timestamp() - case[0]["time:timestamp"].timestamp() for case in log])
        time_between_events_max = np.max(
            [event["time:timestamp"].timestamp() - case[event_index - 1]["time:timestamp"].timestamp() for case in
             log for event_index, event in enumerate(case) if event_index > 0])
        log_time = np.max([event["time:timestamp"].timestamp() for case in log for event in case]) - log[0][0]["time:timestamp"].timestamp()
        self.time_scaling_divisor = [cycle_time_max, time_between_events_max, 86400, 604800, 31536000, log_time]

        # Event dimension: Maximum number of events in a case
        self.event_dim = _get_max_case_length(log)

        # Feature dimension: Encoding size of an event
        activity_encoding_length = len(self.activities)
        time_encoding_length = 6
        categorical_attributes_encoding_length = sum([len(values) for values in self.categorical_attributes_values])
        numerical_attributes_encoding_length = len(self.numerical_attributes)
        text_encoding_length = self.text_encoder.encoding_length if self.text_encoder else 0
        self.feature_dim = self.feature_dim = activity_encoding_length + time_encoding_length + categorical_attributes_encoding_length + numerical_attributes_encoding_length + text_encoding_length

        # Train text encoder
        if self.text_encoder is not None and self.text_attribute is not None:
            docs = [event[self.text_attribute] for case in log for event in case]
            self.text_encoder.fit(docs)

    def transform(self, log, for_training=True):
        case_dim = np.sum([len(case) for case in log]) if for_training else len(log)

        # Prepare input and output vectors/matrices
        x = np.zeros((case_dim, self.event_dim, self.feature_dim))
        if for_training:
            y_next_act = np.zeros((case_dim, len(self.activities) + 1))
            y_final_act = np.zeros((case_dim, len(self.activities)))
            y_next_time = np.zeros(case_dim)
            y_final_time = np.zeros(case_dim)

        # Encode traces and prefix traces
        trace_dim_index = 0
        for case in log:
            case_start_time = case[0]["time:timestamp"].timestamp()
            # For training: Encode all prefixes. For predicting: Encode given prefix only
            prefix_lengths = range(1, len(case) + 1) if for_training else range(len(case), len(case) + 1)
            for prefix_length in prefix_lengths:
                # Encode the (prefix-)trace
                previous_event_time = case_start_time
                # Post padding of event sequences
                padding = self.event_dim - prefix_length
                for event_index, event in enumerate(case):

                    if event_index <= prefix_length - 1:
                        # Encode activity
                        x[trace_dim_index][padding+event_index][self.activities.index(event["concept:name"])] = 1
                        offset = len(self.activities)

                        # Encode time attributes
                        event_time = event["time:timestamp"]
                        # Seconds since case start
                        x[trace_dim_index][padding+event_index][offset] = (event_time.timestamp() - case_start_time)/self.time_scaling_divisor[0]
                        # Seconds since previous event
                        x[trace_dim_index][padding+event_index][offset + 1] = (event_time.timestamp() - previous_event_time)/self.time_scaling_divisor[1]
                        # Seconds since midnight
                        x[trace_dim_index][padding+event_index][offset + 2] = (event_time.timestamp() - event_time.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())/self.time_scaling_divisor[2]
                        # Seconds since last Monday
                        x[trace_dim_index][padding+event_index][offset + 3] = (event_time.weekday() * 86400 + event_time.hour * 3600 + event_time.second)/self.time_scaling_divisor[3]
                        # Seconds since last Januar 1
                        x[trace_dim_index][padding+event_index][offset + 4] = (event_time.timestamp() - event_time.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0).timestamp())/self.time_scaling_divisor[4]
                        # Seconds since process start
                        x[trace_dim_index][padding+event_index][offset + 5] = (event_time.timestamp() - self.process_start_time)/self.time_scaling_divisor[5]

                        previous_event_time = event_time.timestamp()
                        offset += 6

                        # Encode categorical attributes
                        for attribute_index, attribute in enumerate(self.categorical_attributes):
                            x[trace_dim_index][padding+event_index][
                                offset + self.categorical_attributes_values[attribute_index].index(event[attribute])] = 1
                            offset += len(self.categorical_attributes_values[attribute_index])

                        # Encode numerical attributes
                        for attribute_index, attribute in enumerate(self.numerical_attributes):
                            x[trace_dim_index][padding+event_index][offset] = float(event[attribute])
                            offset += 1

                        # Encode textual attribute
                        if self.text_encoder is not None and self.text_attribute is not None:
                            text_vectors = self.text_encoder.transform([event[self.text_attribute]])
                            x[trace_dim_index][padding+event_index][offset:offset+self.text_encoder.encoding_length] = text_vectors[0]
                            offset += self.text_encoder.encoding_length

                # Set activity and time (since case start) of next event as target
                if for_training:
                    if prefix_length == len(case):
                        # Case 1: Set <Process end> as next activity target
                        y_next_act[trace_dim_index][len(self.activities)] = 1
                        y_next_time[trace_dim_index] = (case[-1]["time:timestamp"].timestamp() - case_start_time) / self.time_scaling_divisor[0]
                    else:
                        # Case 2: Set next activity as target
                        y_next_act[trace_dim_index][self.activities.index(case[prefix_length]["concept:name"])] = 1
                        y_next_time[trace_dim_index] = (case[prefix_length]["time:timestamp"].timestamp() - case_start_time) / self.time_scaling_divisor[0]
                    # Set final activity and case cycle time as target
                    y_final_act[trace_dim_index][self.activities.index(case[-1]["concept:name"])] = 1
                    y_final_time[trace_dim_index] = (case[-1]["time:timestamp"].timestamp() - case_start_time) / self.time_scaling_divisor[0]

                # Increase index for next (prefix-)trace
                trace_dim_index += 1

        if for_training:
            return x, y_next_act, y_final_act, y_next_time, y_final_time
        else:
            return x


def _get_event_labels(log, attribute_name):
    return list(dict.fromkeys([event[attribute_name] for case in log for event in case])) if log else []


def _is_numerical(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def _is_numerical_attribute(log, attribute):
    return _is_numerical(log[0][0][attribute])


def _get_max_case_length(log):
    return max([len(case) for case in log]) if log else 0
