import tensorflow as tf
from tensorflow.keras import layers
from .layers import GEGLU, LocalMHA
# Define FeedForward layer
def FeedForward(dim, mult=4, dropout=0.):
    inner_dim = int(dim * mult * 2 / 3)

    return tf.keras.Sequential([
        tf.keras.layers.LayerNormalization(epsilon=1e-6),
        tf.keras.layers.Dense(inner_dim * 2, use_bias=False),
        GEGLU(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(dim, use_bias=False)
    ])

def LocalTransformer(
    num_tokens,
    dim,
    depth,
    local_attn_window_size=512,
    dim_head=64,
    heads=8,
    ff_mult=4,
    attn_dropout=0.0,
    ff_dropout=0.0,
    **kwargs
):
    
    def _apply(x):
        
        for _ in range(depth):
            attn = LocalMHA(
                local_attn_window_size,
                dim_head=dim_head,
                heads=heads,
                dropout=attn_dropout,
                prenorm=True,
                **kwargs
            )

            ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            x_skip = x
            x = attn(x)
            x = layers.Add()([x_skip, x]) 
            x_skip = x
            x = ff(x)
            x = layers.Add()([x_skip, x]) 

        logits = layers.Dense(num_tokens, use_bias=False)(x)
        logits = layers.LayerNormalization()(logits)
        return logits

    return _apply


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
