import tensorflow as tf
from tensorflow.keras import layers
from .layers import GEGLU, LocalMHA
# Define FeedForward layer
def FeedForward(dim, mult=4, dropout=0.):
    """
    A feedforward layer composed of a dense layer with gated linear unit (GLU) activation, dropout, and layer normalization.
    
    Args:
        dim: An integer representing the dimensionality of the input and output tensors.
        mult: A float representing the multiplier to compute the inner dimension of the dense layer. Defaults to 4.
        dropout: A float representing the dropout rate. Defaults to 0.
        
    Returns:
        A Keras Sequential model composed of a layer normalization layer, a dense layer with GLU activation, dropout, and a dense layer.
    """

    inner_dim = int(dim * mult * 2 / 3)

    return tf.keras.Sequential([
        tf.keras.layers.LayerNormalization(epsilon=1e-6),
        tf.keras.layers.Dense(inner_dim * 2, use_bias=False),
        GEGLU(),
        tf.keras.layers.Dropout(dropout),
        tf.keras.layers.Dense(dim, use_bias=False)
    ])

def LocalTransformer(
    dim,
    depth,
    out_dim=None,
    local_attn_window_size=512,
    dim_head=64,
    heads=8,
    ff_mult=4,
    attn_dropout=0.0,
    ff_dropout=0.0,
    **kwargs
):
    """
    A function that returns a callable that applies a local transformer on the input tensor.

    Args:
        dim: An integer, the dimension of the input tensor.
        depth: An integer, the depth of the local transformer.
        out_dim: An integer, the dimension of the output tensor. Defaults to dim.
        local_attn_window_size: An integer, the size of the local attention window. Defaults to 512.
        dim_head: An integer, the dimension of each attention head. Defaults to 64.
        heads: An integer, the number of attention heads. Defaults to 8.
        ff_mult: An integer, the multiplier for the hidden dimension of the feedforward network. Defaults to 4.
        attn_dropout: A float, the dropout rate for the attention layer. Defaults to 0.0.
        ff_dropout: A float, the dropout rate for the feedforward layer. Defaults to 0.0.
        **kwargs: Additional keyword arguments passed to the LocalMHA layer.

    Returns:
        A callable that applies a local transformer on the input tensor.
    """
    if not out_dim:
        out_dim = dim
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

        logits = layers.Dense(out_dim, use_bias=False)(x)
        logits = layers.LayerNormalization()(logits)
        return logits
    return _apply


def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
