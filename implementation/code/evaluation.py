from pm4py.objects.log.importer.xes import factory as xes_import_factory
from tapp_model import TappModel
from log_encoder import LogEncoder
from text_encoder import BoWTextEncoder
from text_encoder import BoNGTextEncoder
from text_encoder import PVTextEncoder
from text_encoder import LDATextEncoder

# Load event data
path_to_data = "../logs/"
file_name = "hospital_billing_filtered.xes"
path = path_to_data + file_name
log = xes_import_factory.apply(path, variant="nonstandard", parameters={"timestamp_sort": True})
split = len(log) // 3 * 2
train_log = log[:split]
test_log = log[split:]

# Configure and build model
language = "english"
text_models = [
    BoWTextEncoder(encoding_length=100, language=language),
    BoNGTextEncoder(n=2, encoding_length=100, language=language),
    PVTextEncoder(encoding_length=20, language=language),
    LDATextEncoder(encoding_length=10, language=language)
]

log_encoder = LogEncoder(text_encoder=text_models[0])
model = TappModel(log_encoder=log_encoder, num_shared_layer=1, num_specialized_layer=1, neurons_per_layer=100)

# Fit and evaluate

history = model.fit(train_log, data_attributes=["age", "gender", "theme"], text_attribute="question")
model.evaluate(test_log)
