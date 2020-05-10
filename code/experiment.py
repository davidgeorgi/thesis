from pm4py.objects.log.importer.xes import factory as xes_import_factory
from prediction_engine import PredictionEngine
from model import LSTMModel
from log_encoder import LSTMLogEncoder
from text_encoder import BoNGTextEncoder


# Load event data
path_to_data = "../data/logs/"
file_name = "hospital_billing_filtered.xes"
path = path_to_data + file_name
log = xes_import_factory.apply(path, variant="nonstandard", parameters={"timestamp_sort": True})

# Choose encoding and model
text_encoder = BoNGTextEncoder(n=2, encoding_length=50, language="english")
log_encoder = LSTMLogEncoder(text_encoder=text_encoder)
model = LSTMModel(log_encoder=log_encoder)

# Build and train model
predictionEngine = PredictionEngine(model=model)
history = predictionEngine.fit(log, data_attributes=[], text_attribute=None)
