from tensorflow.keras import layers
from tensorflow import keras
import tensorflow.keras.backend as K
import tensorflow as tf
from .blocks import LocalTransformer, mlp


class AudioEmbedding(layers.Layer):
    def __init__(self, num_hid=64, maxlen=100):
        super().__init__()
        self.conv1 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv2 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.conv3 = tf.keras.layers.Conv1D(
            num_hid, 11, strides=2, padding="same", activation="relu"
        )
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=num_hid)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return self.conv3(x)
     
def RawAudioClassifier( maxlen=16000, local_attn_window_size=40, dim_head=64, depth=2, mlp_head_units=[64, 32],  projection_dim=64, num_heads=8, num_classes=100):
    inputs = keras.layers.Input(shape=(maxlen, 1))
    encoded_audio = AudioEmbedding(num_hid=projection_dim, maxlen=maxlen)(inputs)

    encoded_audio = LocalTransformer( projection_dim, depth, 
                        local_attn_window_size=local_attn_window_size, dim_head=dim_head, heads=num_heads)(encoded_audio)
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_audio)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)                     
    return keras.Model(inputs=inputs, outputs=logits)
