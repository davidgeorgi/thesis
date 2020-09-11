from abc import ABC, abstractmethod


class PredictionModel(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, log):
        pass

    @abstractmethod
    def predict_next_activity(self, log):
        pass

    @abstractmethod
    def predict_outcome(self, log):
        pass

    @abstractmethod
    def predict_next_time(self, log):
        pass

    @abstractmethod
    def predict_cycle_time(self, log):
        pass

    @abstractmethod
    def predict(selfself, log):
        pass

    @abstractmethod
    def predict_suffix(self, log):
        pass

    @abstractmethod
    def evaluate(self, log):
        pass
