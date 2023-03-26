
from tensorflow import keras
from tensorflow.keras import layers
from .layers import TokenAndPositionEmbedding
from .blocks import LocalTransformer, mlp
def TextClassifier(maxlen, vocab_size, num_classes, embed_dim=32, depth=3, local_attn_window_size=8, mlp_head_units = [2048, 1024]):
    emb_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)

    inputs = layers.Input(shape=(maxlen,))
    audio_emb = emb_layer(inputs)
    representation = LocalTransformer(
        maxlen,
        embed_dim,
        depth,
        local_attn_window_size = local_attn_window_size)(audio_emb)
    representation = layers.GlobalAveragePooling1D()(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    outputs = layers.Dense(num_classes)(features)

    return keras.Model(inputs=inputs, outputs=outputs)
