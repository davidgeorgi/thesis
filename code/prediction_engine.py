

class PredictionEngine:

    def __init__(self, model):
        self.model = model
        self.activites = []

    def fit(self, log, data_attributes=None, text_attribute=None):

        # Build prediction model
        self.activites = _get_event_labels(log, "concept:name")
        return self.model.fit(log, activities=self.activites, data_attributes=data_attributes, text_attribute=text_attribute)

    def predict(self, log):
        pass


def _get_event_labels(log, attribute_name):
    return list(dict.fromkeys([event[attribute_name] for case in log for event in case])) if log else []