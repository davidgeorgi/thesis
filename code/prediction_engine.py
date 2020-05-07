import tensorflow as tf
from tensorflow.keras import metrics
from tensorflow.keras import layers


class PredictionEngine:

    def __init__(self, log, log_encoder):
        self.log = log
        self.log_encoder = log_encoder
        self.activities = log_encoder.activities
        self.model = None

    def build_lstm_model(self, timesteps, feature_dim, num_shared_layer=1, num_specialized_layer=1, neurons_per_layer=100):
        # Input layer
        inputs = tf.keras.Input(shape=(timesteps, feature_dim),  name='inputs')

        # Shared layer
        previous_layer = inputs
        for layer in range(num_shared_layer):
            shared_lstm_layer = layers.LSTM(neurons_per_layer, implementation=2, kernel_initializer='glorot_uniform', return_sequences=True, dropout=0.2)(previous_layer)
            shared_lstm_normalization_layer = layers.BatchNormalization()(shared_lstm_layer)
            previous_layer = shared_lstm_normalization_layer

        previous_layer_next_activity = previous_layer
        previous_layer_final_activity = previous_layer
        previous_layer_next_timestamp = previous_layer
        previous_layer_final_timestamp = previous_layer
        for layer in range(num_specialized_layer):
            next_activity_lstm_layer = layers.LSTM(neurons_per_layer, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(previous_layer_next_activity)
            final_activity_lstm_layer = layers.LSTM(neurons_per_layer, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(previous_layer_final_activity)
            next_timestamp_lstm_layer = layers.LSTM(neurons_per_layer, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(previous_layer_next_timestamp)
            final_timestamp_lstm_layer = layers.LSTM(neurons_per_layer, implementation=2, kernel_initializer='glorot_uniform', return_sequences=False, dropout=0.2)(previous_layer_final_timestamp)
            next_activity_normalization_layer = layers.BatchNormalization()(next_activity_lstm_layer)
            final_activity_normalization_layer = layers.BatchNormalization()(final_activity_lstm_layer)
            next_timestamp_normalization_layer = layers.BatchNormalization()(next_timestamp_lstm_layer)
            final_timestamp_normalization_layer = layers.BatchNormalization()(final_timestamp_lstm_layer)
            previous_layer_next_activity = next_activity_normalization_layer
            previous_layer_final_activity = final_activity_normalization_layer
            previous_layer_next_timestamp = next_timestamp_normalization_layer
            previous_layer_final_timestamp = final_timestamp_normalization_layer

        # Output layer
        next_activity_output = layers.Dense(len(self.activities) + 1, activation='softmax', kernel_initializer='glorot_uniform', name='next_activity_output')(previous_layer_next_activity)
        final_activity_output = layers.Dense(len(self.activities), activation='softmax', kernel_initializer='glorot_uniform', name='final_activity_output')(previous_layer_final_activity)
        next_timestamp_ouput = layers.Dense(1, kernel_initializer='glorot_uniform', name='next_timestamp_output')(previous_layer_next_timestamp)
        final_timestamp_ouput = layers.Dense(1, kernel_initializer='glorot_uniform', name='final_timestamp_output')(previous_layer_final_timestamp)

        # Build and configure model
        model = tf.keras.Model(inputs=[inputs], outputs=[next_activity_output, final_activity_output, next_timestamp_ouput, final_timestamp_ouput])
        optimizer_param = tf.keras.optimizers.Nadam(learning_rate=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-07, name='Nadam')
        metric_param = {'next_activity_output':metrics.CategoricalAccuracy(), 'final_activity_output':metrics.CategoricalAccuracy(), 'next_timestamp_output':metrics.MeanAbsoluteError(), 'final_timestamp_output': metrics.MeanAbsoluteError()}
        loss_param = {'next_activity_output':'categorical_crossentropy', 'final_activity_output':'categorical_crossentropy', 'next_timestamp_output':'mae', 'final_timestamp_output':'mae'}
        model.compile(loss=loss_param, metrics=metric_param, optimizer=optimizer_param)
        return model

    def fit(self, epochs=100):
        x, y_next_act, y_final_act, y_next_time, y_final_time = self.log_encoder.encode(self.log)
        self.model = self.build_lstm_model(x.shape[1], x.shape[2])
        return self.model.fit(x, {'next_activity_output': y_next_act, 'final_activity_output': y_final_act, 'next_timestamp_output': y_next_time, 'final_timestamp_output': y_final_time}, epochs=epochs, batch_size=None, validation_split=0.2)

    def predict(self, log):
        pass


