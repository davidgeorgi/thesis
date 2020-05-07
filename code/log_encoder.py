import numpy as np


class LogEncoder:
    
    def __init__(self, log, data_attributes=[], text_attribute=None, text_encoder=None):
        self.activities = _get_event_labels(log, "concept:name")
        self.data_attributes = data_attributes
        self.text_attribute = text_attribute
        self.text_encoder = text_encoder
        self.categorical_attributes = list(filter(lambda attribute: not _is_numerical_attribute(log, attribute), data_attributes))
        self.categorical_attributes_values = [_get_event_labels(log, attribute) for attribute in self.categorical_attributes]
        self.numerical_attributes = list(filter(lambda attribute: _is_numerical_attribute(log, attribute), data_attributes))
        self.event_dim = _get_max_case_length(log)

        #   Feature dimension: Encoding size of an event
        # = Number of activities (1-hot)
        # + 6 features for time
        # + Size of encodings of additional categorical attributes (1-hot)
        # + Number of additional numerical attributes
        # + Size of text encoding
        self.feature_dim = self.feature_dim = len(self.activities) + 6 + sum([len(values) for values in self.categorical_attributes_values]) + len(self.numerical_attributes) + 0

    def encode(self, log):
        case_dim = np.sum([len(case) for case in log])

        # Prepare input and output vectors/matrices
        x = np.zeros((case_dim, self.event_dim, self.feature_dim))
        y_next_act = np.zeros((case_dim, len(self.activities) + 1))
        y_final_act = np.zeros((case_dim, len(self.activities)))
        y_next_time = np.zeros(case_dim)
        y_final_time = np.zeros(case_dim)

        # Scaling divisors for time related features to achieve values between 0 and 1
        cycle_time_max = np.max(
            [case[-1]["time:timestamp"].timestamp() - case[0]["time:timestamp"].timestamp() for case in log])
        time_between_events_max = np.max(
            [event["time:timestamp"].timestamp() - case[event_index - 1]["time:timestamp"].timestamp() for case in
             log for event_index, event in enumerate(case) if event_index > 0])
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
                        # Encode activity
                        x[trace_dim_index][event_index][self.activities.index(event["concept:name"])] = 1
                        offset = len(self.activities)

                        # Encode time attributes
                        event_time = event["time:timestamp"]
                        # Seconds since case start
                        x[trace_dim_index][event_index][offset] = (event_time.timestamp() - case_start_time)/time_scaling_divisor[0]
                        # Seconds since previous event
                        x[trace_dim_index][event_index][offset + 1] = (event_time.timestamp() - previous_event_time)/time_scaling_divisor[1]
                        # Seconds since midnight
                        x[trace_dim_index][event_index][offset + 2] = (event_time.timestamp() - event_time.replace(hour=0, minute=0, second=0, microsecond=0).timestamp())/time_scaling_divisor[2]
                        # Seconds since last Monday
                        x[trace_dim_index][event_index][offset + 3] = (event_time.weekday() * 86400 + event_time.hour * 3600 + event_time.second)/time_scaling_divisor[3]
                        # Seconds since last Januar 1
                        x[trace_dim_index][event_index][offset + 4] = (event_time.timestamp() - event_time.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0).timestamp())/time_scaling_divisor[4]
                        # Seconds since process start
                        x[trace_dim_index][event_index][offset + 5] = (event_time.timestamp() - process_start_time)/time_scaling_divisor[5]

                        previous_event_time = event_time.timestamp()
                        offset += 6

                        # Encode categorical attributes
                        for attribute_index, attribute in enumerate(self.categorical_attributes):
                            x[trace_dim_index][event_index][
                                offset + self.categorical_attributes_values[attribute_index].index(event[attribute])] = 1
                            offset += len(self.categorical_attributes_values[attribute_index])

                        # Encode numerical attributes
                        for attribute_index, attribute in enumerate(self.numerical_attributes):
                            x[trace_dim_index][event_index][offset] = float(event[attribute])
                            offset += 1

                        # Encode textual attribute

                # Set activity and time (since case start) of next event as target
                if prefix_length == len(case):
                    # Case 1: Set <Process end> as next activity target
                    y_next_act[trace_dim_index][len(self.activities)] = 1
                    y_next_time[trace_dim_index] = (case[-1]["time:timestamp"].timestamp() - case_start_time) / time_between_events_max
                else:
                    # Case 2: Set next activity as target
                    y_next_act[trace_dim_index][self.activities.index(case[prefix_length]["concept:name"])] = 1
                    y_next_time[trace_dim_index] = (case[prefix_length]["time:timestamp"].timestamp() - case_start_time) / time_between_events_max
                # Set final activity and case cycle time as target
                y_final_act[trace_dim_index][self.activities.index(case[-1]["concept:name"])] = 1
                y_final_time[trace_dim_index] = (case[-1]["time:timestamp"].timestamp() - case_start_time) / cycle_time_max

                # Increase index for next (prefix-)trace
                trace_dim_index += 1

        return x, y_next_act, y_final_act, y_next_time, y_final_time


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