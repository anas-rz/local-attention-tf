
from tensorflow import keras
from tensorflow.keras import layers
from .layers import TokenAndPositionEmbedding
from .blocks import LocalTransformer
def TextClassifier(maxlen, vocab_size, num_classes, embed_dim=32, depth=3, local_attn_window_size=8):
    emb_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)

    inputs = layers.Input(shape=(maxlen,))
    x = emb_layer(inputs)
    x = LocalTransformer(
        maxlen,
        embed_dim,
        depth,
        local_attn_window_size = local_attn_window_size)(x)
    x = layers.GlobalAveragePooling1D()(x)
    if num_classes == 1:
        outputs = layers.Dense(num_classes, activation="sigmoid")(x)
    else:
        outputs = layers.Dense(num_classes, activation="softmax")(x)

    return keras.Model(inputs=inputs, outputs=outputs)
