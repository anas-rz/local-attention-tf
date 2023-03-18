from tensorflow.keras import layers
from tensorflow import keras

def LocalMHA( input_shape,
        window_size,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        prenorm = False,
        qk_rmsnorm = False,
        qk_scale = 8,
        use_xpos = False,
        xpos_scale_base = None):
    dim = input_shape[-1]
    inner_dim = dim_head * heads
    i_p = layers.Input(shape=input_shape)
    if prenorm:
        i_p = layers.LayerNormalization()(i_p)
    x = layers.Dense(inner_dim * 3) (i_p)
    q, k, v = layers.Lambda(lambda _x: tf.split(_x, 3, axis=-1))(x)
    q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = heads), (q, k, v))
    attn = LocalAttention(window_size)(q, k, v)
    # attn = rearrange(attn, 'b h n d -> b n (h d)')
    out = layers.Dense(dim)(attn)
    return keras.Model(i_p, out)
