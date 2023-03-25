import tensorflow as tf
from tensorflow.keras import layers

# Define GEGLU layer
class GEGLU(tf.keras.layers.Layer):
    def call(self, x):
        x, gate = tf.split(x, num_or_size_splits=2, axis=-1)
        return x * tf.nn.gelu(gate)

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

class DynamicPositionBias(tf.keras.layers.Layer):
    def __init__(
        self,
        dim,
        heads
    ):
        super().__init__()
        self.mlp = tf.keras.Sequential([
            tf.keras.layers.Dense(dim, activation=tf.nn.silu),
            tf.keras.layers.Dense(dim, activation=tf.nn.silu),
            tf.keras.layers.Dense(heads)
        ])

    def call(self, inputs):
        i, j = inputs
        assert j >= i

        rel_dist = tf.range(j, dtype=tf.float32)
        bias = self.mlp(tf.expand_dims(rel_dist, axis=-1))
        bias = tf.transpose(bias, perm=[1, 0])

        i_seq = tf.range(j - i, j)
        j_seq = tf.range(j)
        rel_dist_indices = tf.abs(tf.expand_dims(i_seq, axis=-1) - tf.expand_dims(j_seq, axis=0))

        bias = tf.gather(bias, rel_dist_indices)
        bias = tf.transpose(bias, perm=[2, 0, 1])
        return bias

def LocalTransformer(
    num_tokens,
    max_seq_len,
    dim,
    depth,
    causal=True,
    local_attn_window_size=512,
    dim_head=64,
    heads=8,
    ff_mult=4,
    attn_dropout=0.0,
    ff_dropout=0.0,
    ignore_index=-1,
    use_xpos=False,
    xpos_scale_base=None,
    use_dynamic_pos_bias=False,
    **kwargs
):
    token_emb = layers.Embedding(num_tokens, dim)
    pos_emb = layers.Embedding(max_seq_len, dim)

    inputs = layers.Input(shape=(max_seq_len,))
    x = token_emb(inputs)

    n = max_seq_len
    attn_bias = None
    if use_dynamic_pos_bias:
        w = local_attn_window_size
        dynamic_pos_bias = DynamicPositionBias(dim=dim // 2, heads=heads)
        attn_bias = dynamic_pos_bias(w, w * 2)

    for _ in range(depth):
        attn = LocalMHA(
            local_attn_window_size,
            dim_head=dim_head,
            heads=heads,
            dropout=attn_dropout,
            causal=causal,
            use_xpos=use_xpos,
            xpos_scale_base=xpos_scale_base,
            use_rotary_pos_emb=not use_dynamic_pos_bias,
            prenorm=True,
            **kwargs
        )

        ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)

        x = attn(x, mask=None, attn_bias=attn_bias) + x

        x = ff(x) + x

    logits = layers.Dense(num_tokens, use_bias=False)(x)
    logits = layers.LayerNormalization()(logits)

    return tf.keras.Model(inputs=inputs, outputs=logits)
