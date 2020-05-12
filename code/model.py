import numpy as np
import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from abc import ABC, abstractmethod


class Model(ABC):

    def __init__(self):
        super().__init__()

    @abstractmethod
    def fit(self, log):
        pass

    @abstractmethod
    def predict_next(self, log):
        pass

    def predict_final(self, log):
        pass


class LSTMModel(Model):

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
            shared_lstm_normalization_layer = layers.BatchNormalization()(shared_lstm_layer)
            previous_layer = shared_lstm_normalization_layer

        previous_layer_next_activity = previous_layer
        previous_layer_final_activity = previous_layer
        previous_layer_next_timestamp = previous_layer
        previous_layer_final_timestamp = previous_layer
        for layer in range(self.num_specialized_layer):
            next_activity_lstm_layer = layers.LSTM(self.neurons_per_layer, implementation=2, kernel_initializer="glorot_uniform", return_sequences=False, dropout=0.2)(previous_layer_next_activity)
            final_activity_lstm_layer = layers.LSTM(self.neurons_per_layer, implementation=2, kernel_initializer="glorot_uniform", return_sequences=False, dropout=0.2)(previous_layer_final_activity)
            next_timestamp_lstm_layer = layers.LSTM(self.neurons_per_layer, implementation=2, kernel_initializer="glorot_uniform", return_sequences=False, dropout=0.2)(previous_layer_next_timestamp)
            final_timestamp_lstm_layer = layers.LSTM(self.neurons_per_layer, implementation=2, kernel_initializer="glorot_uniform", return_sequences=False, dropout=0.2)(previous_layer_final_timestamp)
            next_activity_normalization_layer = layers.BatchNormalization()(next_activity_lstm_layer)
            final_activity_normalization_layer = layers.BatchNormalization()(final_activity_lstm_layer)
            next_timestamp_normalization_layer = layers.BatchNormalization()(next_timestamp_lstm_layer)
            final_timestamp_normalization_layer = layers.BatchNormalization()(final_timestamp_lstm_layer)
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

    def fit(self, log, data_attributes=None, text_attribute=None, epochs=3):

        # Encode training data
        self.activities = _get_event_labels(log, "concept:name")
        self.log_encoder.fit(log, activities=self.activities, data_attributes=data_attributes, text_attribute=text_attribute)
        x, y_next_act, y_final_act, y_next_time, y_final_time = self.log_encoder.transform(log, for_training=True)

        # Build model
        self.timesteps = x.shape[1]
        self.feature_dim = x.shape[2]
        self._build_model()

        # Reduce learning rate if metrics do not improve anymore
        reduce_lr = ReduceLROnPlateau(monitor="val_loss", factor=0.1, patience=6, min_lr=0.0001)
        # Stop Early if metrics do not improve for longer time
        early_stopping = EarlyStopping(monitor="val_loss", patience=20)
        # Fit the model to data
        return self.model.fit(x, {"next_activity_output": y_next_act, "final_activity_output": y_final_act, "next_timestamp_output": y_next_time, "final_timestamp_output": y_final_time}, epochs=epochs, batch_size=None, validation_split=0.2, callbacks=[reduce_lr, early_stopping])

    def predict_next_activity(self, log):
        x = self.log_encoder.transform(log, for_training=False)
        pred = self.model.predict(x)
        return np.array([np.argmax(act) for act in pred[0]])

    def predict_final_activity(self, log):
        x = self.log_encoder.transform(log, for_training=False)
        pred = self.model.predict(x)
        return np.array([np.argmax(act) for act in pred[1]])

    def predict_next_time(self, log):
        x = self.log_encoder.transform(log, for_training=False)
        pred = self.model.predict(x)
        return pred[2].flatten() * self.log_encoder.time_scaling_divisor[1]

    def predict_final_time(self, log):
        x = self.log_encoder.transform(log, for_training=False)
        pred = self.model.predict(x)
        return pred[3].flatten() * self.log_encoder.time_scaling_divisor[0]

    def _get_activity_label(self, index):
        return self.activities[index]


def _get_event_labels(log, attribute_name):
    return list(dict.fromkeys([event[attribute_name] for case in log for event in case])) if log else []
