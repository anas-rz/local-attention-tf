import tensorflow as tf
from tensorflow.keras import layers
from einops import rearrange
import tensorflow.keras.backend as K

def default(value, d):
    return d if not exists(value) else value

def exists(val):
    return val is not None

@tf.function
def look_around(x, backward=1, forward=0, pad_value=-1, axis=2):
    t = K.int_shape(x)[1]
    tensor_rank = tf.rank(x)
    padding_tensor = tf.zeros((tensor_rank, 2), dtype=tf.int32)
    padding_tensor = tf.tensor_scatter_nd_update(padding_tensor, [[1, 0], [1, 1]], [backward, forward])
    padded_x = tf.pad(x, padding_tensor, constant_values=pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return tf.concat(tensors, axis=axis)



class LocalAttention(layers.Layer):
    def __init__(
        self,
        window_size,
        look_backward = 1,
        look_forward = None,
        dropout = 0.,
    ):
        super().__init__()
        look_forward = 0

        self.window_size = window_size

        self.look_backward = look_backward
        self.look_forward = look_forward

        self.dropout = layers.Dropout(dropout)


    def call(
        self,
        q, k, v,
        window_size = None,
    ):
        pad_value, window_size,  look_backward, look_forward =  -1, default(window_size, self.window_size), self.look_backward, self.look_forward
        h_q, n_q, dim_q = K.int_shape(q)[1], K.int_shape(q)[2], K.int_shape(q)[3]
        q, k, v = map(lambda t: rearrange(t, 'b h n d -> (b h) n d'), (q, k, v))
        b, n, dim_head = K.int_shape(q)[0], K.int_shape(q)[1], K.int_shape(q)[2]
        scale = dim_head ** -0.5
        assert (n % window_size) == 0, f'sequence length {n} must be divisible by window size {window_size} for local attention'
        windows = n // window_size
        # bucketing
        bq, bk, bv = map(lambda t: rearrange(t, 'b (w n) d -> b w n d', w = windows), (q, k, v))
        bq = bq * scale
        look_around_kwargs = dict(
            backward =  look_backward,
            forward =  look_forward,
            pad_value = pad_value
        )
        bk = look_around(bk, **look_around_kwargs)
        bv = look_around(bv, **look_around_kwargs)
        sim = tf.einsum('b h i e, b h j e -> b h i j', bq, bk)
        # attention
        attn = tf.nn.softmax(sim, axis=-1)
        attn = self.dropout(attn)
        # aggregation
        sim = tf.einsum('b h i j, b h j e -> b h i e', attn, bv)
        out = rearrange(sim, 'b w n d -> b (w n) d')
        out = rearrange(out, '(b h) n d -> b h n d', h = h_q)
        return out

class LocalMHA(layers.Layer):
    def __init__(
        self,
        window_size,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        prenorm = False,
        qk_scale = 8,
        
    ):
        super().__init__()        
        self.inner_dim = dim_head * heads
        self.window_size = window_size

        self.heads = heads
        self.prenorm = prenorm

    def build(self, input_shape):
        dim = input_shape[-1]
        self.norm = layers.LayerNormalization(axis=-1) if self.prenorm else None

        self.to_qkv = layers.Dense(self.inner_dim * 3, use_bias = False)


        self.attn_fn = LocalAttention(
            window_size = self.window_size,
        )

        self.to_out = layers.Dense(dim, use_bias = False)

    def call(self, x):
        if exists(self.norm):
            x = self.norm(x)

        q, k, v = tf.split(self.to_qkv(x),3, axis=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v)) 


        out = self.attn_fn(q, k, v)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class TokenAndPositionEmbedding(layers.Layer):
    # https://keras.io/examples/nlp/text_classification_with_transformer/
    def __init__(self, maxlen, vocab_size, embed_dim):
        super().__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions

# Define GEGLU layer
class GEGLU(tf.keras.layers.Layer):
    def call(self, x):
        x, gate = tf.split(x, num_or_size_splits=2, axis=-1)
        return x * tf.nn.gelu(gate)
