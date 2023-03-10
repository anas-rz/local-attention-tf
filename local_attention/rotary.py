import tensorflow as tf
from tensorflow.keras.layers import Layer
from einops import rearrange



class SinusoidalEmbeddings(Layer):
    def __init__(
        self,
        dim,
        scale_base = None,
        use_xpos = False
    ):
        super().__init__()

        self.inv_freq = 1. / (10000 ** tf.cast(tf.range(0, dim, 2), dtype=tf.float32) / dim)

        # xpos related

        self.use_xpos = use_xpos
        self.scale_base = scale_base

        assert not (use_xpos and not exists(scale_base)), 'scale base must be defined if using xpos'

        scale = (tf.range(0, dim, 2, dtype=tf.float32) + 0.4 * dim) / (1.4 * dim)

    def call(self, x):
        seq_len = x.shape[-2]

        t = tf.cast(tf.range(seq_len), self.inv_freq.dtype)
        freqs = tf.einsum('i , j -> i j', t, self.inv_freq)
        freqs =  tf.concat((freqs, freqs), -1)

        if not self.use_xpos:
            return freqs, tf.ones(1)

        power = (t - (seq_len // 2)) / self.scale_base
        scale = self.scale ** rearrange(power, 'n -> n 1')
        scale = tf.concat((scale, scale), -1)

        return freqs, scale


def rotate_half(x):
    x = rearrange(x, 'b ... (r d) -> b ... r d', r = 2)
    x1, x2 = tf.unstack(x, axis = -2)
    return tf.concat((-x2, x1), axis = -1)

def apply_rotary_pos_emb(q, k, freqs, scale = 1):
    q_len = q.shape[-2]
    q_freqs = freqs[..., -q_len:, :]

    inv_scale = scale ** -1

    if len(scale.shape) == 2:
        scale = scale[-q_len:, :]

    q = (q * tf.math.cos(q_freqs) * scale) + (rotate_half(q) * tf.math.sin(q_freqs) * scale)
    k = (k * tf.math.cos(freqs) * inv_scale) + (rotate_half(k) * tf.math.sin(freqs) * inv_scale)
    return q, k
