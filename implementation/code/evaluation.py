from pm4py.objects.log.importer.xes import factory as xes_import_factory
from lstm_model import TappModel
from log_encoder import LogEncoder
from text_encoder import BoNGTextEncoder


# Load event data
path_to_data = "../logs/"
file_name = "hospital_billing_filtered.xes"
path = path_to_data + file_name
log = xes_import_factory.apply(path, variant="nonstandard", parameters={"timestamp_sort": True})

# Choose encoding and model
text_encoder = BoNGTextEncoder(n=2, encoding_length=50, language="english")
log_encoder = LogEncoder(text_encoder=text_encoder)
model = TappModel(log_encoder=log_encoder, num_shared_layer=1, num_specialized_layer=1, neurons_per_layer=100)

# Build and train model
history = model.fit(log, data_attributes=[], text_attribute=None)

model.evaluate(log)
