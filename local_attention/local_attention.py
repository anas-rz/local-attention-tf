import tensorflow as tf
import tensorflow.keras.backend as K
import math
from tensorflow.keras import layers
from einops import rearrange, pack, unpack




def exists(val):
    return val is not None

def default(value, d):
    return d if not exists(value) else value

def max_neg_value(tensor):
    return tensor.dtype.min

def l2norm(tensor):
    dtype = tensor.dtype
    normed = tf.norm(tensor, axis = -1)
    return tf.cast(normed, dtype)

def pad_sequence_to_multiple_of(seq, multiple, axis):

    if axis < 0:
        axis = len(tf.shape(seq)) + axis
    shape = tf.shape(seq)
    length = shape[axis]
    remainder = length % multiple
    padding = tf.cond(tf.equal(remainder, 0), 
                      lambda: 0,
                      lambda: multiple - remainder)
    pad_shape = tf.concat([
        shape[:axis],
        tf.expand_dims(padding, axis=0),
        shape[axis + 1:]], axis=0)
    padding_tensor = tf.zeros(pad_shape, dtype=seq.dtype)
    return tf.concat([seq, padding_tensor], axis=axis)

def look_around(x, backward=1, forward=0, pad_value=-1, axis=2):
    t = K.int_shape(x)[1]
    tensor_rank = tf.rank(x)
    f_i = len(K.int_shape(x)) - axis - 1
    padding_tensor = tf.zeros((tensor_rank, 2), dtype=tf.int32)
    padding_tensor = tf.tensor_scatter_nd_update(padding_tensor, [[f_i, 0], [f_i, 1]], [backward, forward])
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
        exact_windowsize = False,
        scale = None,
    ):
        super().__init__()
        look_forward = 0

        self.scale = scale

        self.window_size = window_size
        self.exact_windowsize = exact_windowsize


        self.look_backward = look_backward
        self.look_forward = look_forward

        self.dropout = layers.Dropout(dropout)


    def call(
        self,
        q, k, v,
        window_size = None
    ):



        shape, pad_value, window_size,  look_backward, look_forward = q.shape, -1, default(window_size, self.window_size), self.look_backward, self.look_forward

        # https://github.com/arogozhnikov/einops/blob/master/docs/4-pack-and-unpack.ipynb
        (q, packed_shape), (k, _), (v, _) = map(lambda t: pack([t], '* n d'), (q, k, v))


        
        b, n, dim_head, dtype = *K.int_shape(q), q.dtype

        scale = default(self.scale, dim_head ** -0.5)

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
        out = tf.einsum('b h i j, b h j e -> b h i e', attn, bv)
        out = rearrange(out, 'b w n d -> b (w n) d')

        out, *_ = unpack(out, packed_shape, '* n d')
        return out
