from tensorflow import keras
from tensorflow.keras import layers
import tensorflow.keras.backend as K
from .blocks import LocalTransformer, mlp
import tensorflow as tf


class Patches(layers.Layer):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(layers.Layer):
    def __init__(self, num_patches, projection_dim):
        super().__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded




def ImageClassifier(img_size=224, patch_size=16, projection_dim=196, depth=6, local_attn_window_size=28,
                    dim_head=196, num_heads=8, num_classes=2, mlp_head_units = [2048, 1024]  # Size of the dense layers of the final classifier
):
    inputs = keras.layers.Input(shape=(img_size, img_size, 3))
    patches = Patches(patch_size)(inputs)
    num_patches = (img_size // patch_size) ** 2
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    encoded_patches = LocalTransformer(num_patches, projection_dim, depth, 
                        local_attn_window_size=local_attn_window_size, dim_head=dim_head, heads=num_heads)(encoded_patches)
    representation = layers.LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = layers.GlobalAveragePooling1D()(representation)
    representation = layers.Dropout(0.5)(representation)
    features = mlp(representation, hidden_units=mlp_head_units, dropout_rate=0.5)
    logits = layers.Dense(num_classes)(features)                     
    return keras.Model(inputs=inputs, outputs=logits)
