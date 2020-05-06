from pm4py.objects.log.importer.xes import factory as xes_import_factory
from prediction_engine import PredictionEngine
from log_encoder import LogEncoder

# Load event data
path_to_data = "../data/logs/"
file_name = "hospital_billing_filtered.xes"
path = path_to_data + file_name
log = xes_import_factory.apply(path, variant="nonstandard", parameters={"timestamp_sort": True})

# Choose encoding
log_encoder = LogEncoder(log, data_attributes=[], text_attribute=None, text_encoder=None)

# Build and train model
predictionEngine = PredictionEngine(log, log_encoder=log_encoder)
history = predictionEngine.fit(epochs=100)
