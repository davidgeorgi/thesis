

class PredictionEngine:

    def __init__(self, model):
        self.model = model

    def fit(self, log, data_attributes=None, text_attribute=None):
        # Build prediction model
        return self.model.fit(log, data_attributes=data_attributes, text_attribute=text_attribute)

    def predict_next(self, log):
        return self.model.predict_next(log)

    def predict_final(self, log):
        return self.model.predict_final(log)

    def evaluate(self, log):
        pass



