

class PredictionEngine:

    def __init__(self, model):
        self.model = model

    def fit(self, log, data_attributes=None, text_attribute=None):
        # Build prediction model
        return self.model.fit(log, data_attributes=data_attributes, text_attribute=text_attribute)

    def predict_next_activity(self, log):
        return self.model.predict_next_activity(log)

    def predict_final_activity(self, log):
        return self.model.predict_final_activity(log)

    def predict_next_time(self, log):
        return self.model.predict_next_time(log)

    def predict_final_time(self, log):
        return self.model.predict_final_time(log)

    def evaluate(self, log):
        pass



