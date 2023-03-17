import tensorflow as tf
import tensorflow.keras.backend as K
import math

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
    f_i = len(K.int_shape(x)) - axis
    padding_tensor = tf.zeros((tensor_rank, 2), dtype=tf.int32)
    padding_tensor = tf.tensor_scatter_nd_update(padding_tensor, [[f_i, 0], [f_i, 1]], [backward, forward])
    padded_x = tf.pad(x, padding_tensor, constant_values=pad_value)
    print(padded_x.shape)
    tensors = [padded_x[:, ind:(ind + t), ...] for ind in range(forward + backward + 1)]
    return tf.concat(tensors, axis=axis)
