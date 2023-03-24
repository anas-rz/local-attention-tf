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
