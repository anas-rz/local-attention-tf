import tensorflow as tf
from tensorflow.keras import layers
from einops import rearrange
import tensorflow.keras.backend as K

def default(value, d):
    """
    The default function takes two arguments: value and d. If the value is None, it returns the default value d, otherwise it returns the value.

    Parameters:

        value: A value to check if it is None or not.
        d: The default value that will be returned if the value is None.
    Returns:

        The value if it is not None, otherwise the default value d.

    """
    return d if not exists(value) else value

def exists(val):
    """
    Checks if a value/module exists.

    Args:
        val: The value/module to check.

    Returns:
        bool: True if the value/module exists, False otherwise.
    """
    return val is not None

@tf.function
def look_around(x, backward=1, forward=0, pad_value=-1, axis=2):
    """
       Applies a sliding window to a tensor along a given axis, and concatenates the resulting tensors.

    Args:
        x: The input tensor.
        backward (int): The number of indices to look backward.
        forward (int): The number of indices to look forward.
        pad_value (int): The padding value to use.
        axis (int): The axis to apply the sliding window along.

    Returns:
        tf.Tensor: The resulting tensor after applying the sliding window and concatenation.
    """
    t = K.int_shape(x)[1]
    tensor_rank = tf.rank(x)
    padding_tensor = tf.zeros((tensor_rank, 2), dtype=tf.int32)
    padding_tensor = tf.tensor_scatter_nd_update(padding_tensor, [[1, 0], [1, 1]], [backward, forward])
    padded_x = tf.pad(x, padding_tensor, constant_values=pad_value)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return tf.concat(tensors, axis=axis)



class LocalAttention(layers.Layer):
    """
    performs local attention on input tensors q, k, and v.

    Args:
        window_size (int): The size of the attention window.
        look_backward (int, optional): The number of previous elements to include in the attention window. Defaults to 1.
        look_forward (int, optional): The number of subsequent elements to include in the attention window. Defaults to None.
        dropout (float, optional): The dropout rate to apply to the attention scores. Defaults to 0.

    Methods:
        call(q, k, v, window_size=None): Performs the local attention operation on the input tensors q, k, and v.

    Returns:
        tf.Tensor: The output tensor after the local attention operation has been performed.
    """
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
    """
    The LocalMHA layer uses a dense linear projection to map the input tensor to separate query, 
    key, and value tensors for each head, which are then passed to the LocalAttention layer to 
    compute the attention scores. The output of the local attention operation is then 
    concatenated across the heads dimension and projected back to the original tensor 
    dimension using another dense layer. 

    Args:
        window_size (int): The size of the local window to apply attention to.
        dim_head (int): The size of each attention head.
        heads (int): The number of attention heads to use.
        dropout (float): The dropout rate to apply to the attention scores.
        prenorm (bool): Whether to apply layer normalization before the attention operation.
        qk_scale (float): The scaling factor to apply to the dot product of the query and key tensors.

    Returns:
        tf.Tensor: The output of the LocalMHA layer, with the same shape as the input tensor.
    """
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
    """
    This layer combines token and position embeddings for input sequences. It takes in a maximum sequence length,
    vocabulary size, and embedding dimension as arguments. It uses two separate embedding layers - one for token
    embeddings and the other for position embeddings. The position embedding is added element-wise to the token
    embedding to create the final embedding.

    Attributes:
        token_emb (tf.keras.layers.Embedding): Embedding layer for token embeddings.
        pos_emb (tf.keras.layers.Embedding): Embedding layer for position embeddings.

    Methods:
        call(x): Takes in an input tensor x and returns the sum of token and position embeddings.
    """
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
    """
    A gated linear unit with exponential linear unit (GEGLU) activation function. This layer splits the input tensor into two halves along the last dimension, applies the gelu activation to the second half, and multiplies the first half by the gelu output.

Arguments:
    None

Call arguments:
    x: Input tensor of shape `(batch_size, ..., input_dim)`. The last dimension must be even.

Returns:
    The GEGLU activation of the input tensor, of shape `(batch_size, ..., input_dim // 2)`.
    """
    def call(self, x):
        x, gate = tf.split(x, num_or_size_splits=2, axis=-1)
        return x * tf.nn.gelu(gate)
