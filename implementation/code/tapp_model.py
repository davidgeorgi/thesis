import numpy as np
import pandas as pd
import tensorflow as tf
from datetime import datetime
import os
from tensorflow.keras import metrics
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from prediction_model import PredictionModel


class TappModel(PredictionModel):

    def __init__(self, log_encoder=None, num_shared_layer=1, num_specialized_layer=1, neurons_per_layer=100):
        self.log_encoder = log_encoder
        self.num_shared_layer = num_shared_layer
        self.num_specialized_layer = num_specialized_layer
        self.neurons_per_layer = neurons_per_layer
        self.activities = []
        self.timesteps = 0
        self.feature_dim = 0
        self.model = None
        super().__init__()

    def _build_model(self):
        # Input layer
        inputs = tf.keras.Input(shape=(self.timesteps, self.feature_dim), name="inputs")

        # Shared layer
        previous_layer = inputs
        for layer in range(self.num_shared_layer):
            shared_lstm_layer = layers.LSTM(self.neurons_per_layer, implementation=2, kernel_initializer="glorot_uniform", return_sequences=True, dropout=0.2)(previous_layer)
            shared_lstm_normalization_layer = layers.LayerNormalization()(shared_lstm_layer)
            previous_layer = shared_lstm_normalization_layer

        previous_layer_next_activity = previous_layer
        previous_layer_final_activity = previous_layer
        previous_layer_next_timestamp = previous_layer
        previous_layer_final_timestamp = previous_layer
        for layer in range(self.num_specialized_layer):
            # Do not return sequences in the last LSTM layer
            return_sequences = False if layer == self.num_specialized_layer-1 else True
            next_activity_lstm_layer = layers.LSTM(self.neurons_per_layer, implementation=2, kernel_initializer="glorot_uniform", return_sequences=return_sequences, dropout=0.2)(previous_layer_next_activity)
            final_activity_lstm_layer = layers.LSTM(self.neurons_per_layer, implementation=2, kernel_initializer="glorot_uniform", return_sequences=return_sequences, dropout=0.2)(previous_layer_final_activity)
            next_timestamp_lstm_layer = layers.LSTM(self.neurons_per_layer, implementation=2, kernel_initializer="glorot_uniform", return_sequences=return_sequences, dropout=0.2)(previous_layer_next_timestamp)
            final_timestamp_lstm_layer = layers.LSTM(self.neurons_per_layer, implementation=2, kernel_initializer="glorot_uniform", return_sequences=return_sequences, dropout=0.2)(previous_layer_final_timestamp)
            next_activity_normalization_layer = layers.LayerNormalization()(next_activity_lstm_layer)
            final_activity_normalization_layer = layers.LayerNormalization()(final_activity_lstm_layer)
            next_timestamp_normalization_layer = layers.LayerNormalization()(next_timestamp_lstm_layer)
            final_timestamp_normalization_layer = layers.LayerNormalization()(final_timestamp_lstm_layer)
            previous_layer_next_activity = next_activity_normalization_layer
            previous_layer_final_activity = final_activity_normalization_layer
            previous_layer_next_timestamp = next_timestamp_normalization_layer
            previous_layer_final_timestamp = final_timestamp_normalization_layer

        # Output layer
        next_activity_output = layers.Dense(len(self.activities) + 1, activation="softmax", kernel_initializer="glorot_uniform", name="next_activity_output")(previous_layer_next_activity)
        final_activity_output = layers.Dense(len(self.activities), activation="softmax", kernel_initializer="glorot_uniform", name="final_activity_output")(previous_layer_final_activity)
        next_timestamp_ouput = layers.Dense(1, kernel_initializer="glorot_uniform", name="next_timestamp_output")(previous_layer_next_timestamp)
        final_timestamp_ouput = layers.Dense(1, kernel_initializer="glorot_uniform", name="final_timestamp_output")(previous_layer_final_timestamp)

        # Build and configure model
        model = tf.keras.Model(inputs=[inputs], outputs=[next_activity_output, final_activity_output, next_timestamp_ouput, final_timestamp_ouput])
        optimizer_param = tf.keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name="Nadam")
        metric_param = {"next_activity_output": metrics.CategoricalAccuracy(), "final_activity_output": metrics.CategoricalAccuracy(), "next_timestamp_output": metrics.MeanAbsoluteError(), "final_timestamp_output": metrics.MeanAbsoluteError()}
        loss_param = {"next_activity_output": "categorical_crossentropy", "final_activity_output": "categorical_crossentropy", "next_timestamp_output": "mae", "final_timestamp_output": "mae"}
        model.compile(loss=loss_param, metrics=metric_param, optimizer=optimizer_param)
        self.model = model

    def fit(self, log, data_attributes=None, text_attribute=None, epochs=100):

        # Encode training data
        self.activities = _get_event_labels(log, "concept:name")
        self.log_encoder.fit(log, activities=self.activities, data_attributes=data_attributes, text_attribute=text_attribute)
        x, y_next_act, y_final_act, y_next_time, y_final_time = self.log_encoder.transform(log, for_training=True)

        # Build model
        self.timesteps = x.shape[1]
        self.feature_dim = x.shape[2]
        self._build_model()

        # Reduce learning rate if metrics do not improve anymore
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=8, min_lr=0.0001)
        # Stop early if metrics do not improve for longer time
        early_stopping = EarlyStopping(monitor="val_loss", patience=20)
        # Save model
        model_checkpoint = ModelCheckpoint("../models/model_{epoch:02d}-{val_loss:.2f}.h5", monitor="val_loss", verbose=0, save_best_only=True, save_weights_only=False, mode="auto")
        # Fit the model to data
        return self.model.fit(x, {"next_activity_output": y_next_act, "final_activity_output": y_final_act, "next_timestamp_output": y_next_time, "final_timestamp_output": y_final_time}, epochs=epochs, batch_size=None, validation_split=0.2, callbacks=[reduce_lr, early_stopping, model_checkpoint])

    def predict_next_activity(self, log):
        x = self.log_encoder.transform(log, for_training=False)
        prediction = self.model.predict(x)
        return prediction[0]

    def predict_final_activity(self, log):
        x = self.log_encoder.transform(log, for_training=False)
        prediction = self.model.predict(x)
        return prediction[1]

    def predict_next_time(self, log):
        x = self.log_encoder.transform(log, for_training=False)
        prediction = self.model.predict(x)
        return prediction[2].flatten() * self.log_encoder.time_scaling_divisor[0]

    def predict_final_time(self, log):
        x = self.log_encoder.transform(log, for_training=False)
        prediction = self.model.predict(x)
        return prediction[3].flatten() * self.log_encoder.time_scaling_divisor[0]

    def predict_suffix(self, log):
        return

    def predict(self, log):
        x = self.log_encoder.transform(log, for_training=False)
        prediction = self.model.predict(x)
        prediction[2] = prediction[2].flatten() * self.log_encoder.time_scaling_divisor[0]
        prediction[3] = prediction[3].flatten() * self.log_encoder.time_scaling_divisor[0]
        return prediction

    def evaluate(self, log):
        folder_path = "../results/lstm-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        os.mkdir(folder_path)

        # Make predictions
        predictions = []
        for case in log:
            caseID = case.attributes["concept:name"]
            for prefix_length in range(1, len(case) + 1):
                prediction = self.predict([case[0:prefix_length]])

                true_next_activity = len(self.activities) if prefix_length == len(case) else self.activities.index(case[prefix_length]["concept:name"])
                true_case_outcome = self.activities.index(case[-1]["concept:name"])
                true_next_time = (case[-1]["time:timestamp"].timestamp() - case[prefix_length - 1]["time:timestamp"].timestamp())/86400 if prefix_length == len(case) else (case[prefix_length]["time:timestamp"].timestamp() - case[prefix_length - 1]["time:timestamp"].timestamp())/86400
                true_cycle_time = (case[-1]["time:timestamp"].timestamp() - case[0]["time:timestamp"].timestamp())/86400

                predicted_next_activity = np.argmax(prediction[0][0])
                predicted_case_outcome = np.argmax(prediction[1][0])
                predicted_next_time = prediction[2][0] / 86400
                predicted_cycle_time = prediction[3][0] / 86400

                predictions.append([caseID, prefix_length, true_next_activity, predicted_next_activity, true_case_outcome, predicted_case_outcome, true_next_time, predicted_next_time, true_cycle_time, predicted_cycle_time])

        # Save predictions in csv file
        columns = ["CaseID", "Prefix length", "True next activity", "Predicted next activity", "True case outcome", "Predicted case outcome", "True next time", "Predicted next time", "True cycle time", "Predicted cylce time"]
        df = pd.DataFrame(predictions, columns=columns)
        df.to_csv(folder_path + "/predictions.csv", encoding='utf-8', sep=',', index=False)
        return df

    def _get_activity_label(self, index):
        return self.activities[index]


def _get_event_labels(log, attribute_name):
    return list(dict.fromkeys([event[attribute_name] for case in log for event in case])) if log else []
