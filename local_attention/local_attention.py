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

def pad_to_multiple(tensor, multiple, axis=-2, value=0):
    assert axis == -2, "Dynamic axis selection not added.."
    seqlen = K.int_shape(tensor)[axis]
    m = seqlen / multiple
    if m.is_integer():
        return False, tensor
    remainder = math.ceil(m) * multiple - seqlen
    paddings = [[0,0], [0, remainder], [0, 0]]
    paddings = tf.convert_to_tensor(paddings,tf.int32)
    return True, tf.pad(tensor, paddings, constant_values = value)
