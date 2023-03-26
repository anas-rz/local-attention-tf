# Local Attention TensorFlow 

![Local Window Attention](https://github.com/lucidrains/local-attention/blob/master/diagram.png?raw=true "Local Window Attention")

[Converted from PyTorch by Phil Wang](https://github.com/lucidrains/local-attention/).

TensorFlow implementation of Local Windowed Attention with classifiers for Language Modeling, Image and Audio. Model codes are xla compatible.


# Usage

## Installation
Make sure to install the latest version

```
pip install local-attention-tf==0.0.3
```
## Using Transformer Layer


```
from local_attention_tf import LocalTransformer
import tensorflow as tf
attn_layer = LocalTransformer(256, 3, local_attn_window_size=128, dim_head=64,
    heads=8,
    ff_mult=4,
    attn_dropout=0.0,
    ff_dropout=0.0,)
inputs = tf.random.uniform((1, 1024, 256))
attended =  attn_layer(inputs)
print(attended.shape) # TensorShape([1, 1024, 256])
```

## Using TextClassifier
The text classifier employing local attention has been trained/tested on IMDB movie review sentiment classification.

```
from local_attention_tf.text import TextClassifier
import tensorflow as tf
model = TextClassifier(maxlen = 200, vocab_size = 20000, num_classes=2, embed_dim=32, depth=3, local_attn_window_size=8, mlp_head_units = [2048, 1024])
dummy_inputs = tf.random.uniform((1, 200))
out =  model(dummy_inputs)
print(out.shape) # TensorShape([1, 2])
```

## Using ImageClassifier
The image classifier employing local attention has been trained/tested on CIFAR 100 dataset.

```
from local_attention_tf.image import ImageClassifier
import tensorflow as tf
model = ImageClassifier(img_size=224, patch_size=16, projection_dim=196, depth=6, local_attn_window_size=28,
                    dim_head=196, num_heads=8, num_classes=2, mlp_head_units = [2048, 1024]  )
dummy_inputs = tf.random.uniform((1, 224, 224, 3))
out =  model(dummy_inputs)
print(out.shape) # TensorShape([1, 2])
```

## Using RawAudioClassifier

```
from local_attention_tf.audio import RawAudioClassifier
import tensorflow as tf
model = RawAudioClassifier(maxlen=16000, local_attn_window_size=40, dim_head=64, depth=2, mlp_head_units=[64, 32],  projection_dim=64, num_heads=8, num_classes=100)
dummy_inputs = tf.random.uniform((1, 16000, 1))
out =  model(dummy_inputs)
print(out.shape) # TensorShape([1, 100])
```

## XLA Support
Inspired from [Sayak Paul](https://github.com/sayakpaul/maxim-tf#xla-support), models have XLA support leading to a significant reduce in latency. See [benchmark_xla.py](./benchmark_xla.py) done for Text Model


To do:

- [X] Implementing rotary module
- [X] Local Attention Module 
- [X] Transformer Module
- [X] Restructure and improve code
- [X] Provide Applied examples for Audio, Image, Text
- [X] Testing of the framework on real world dataset
- [X] Exposing as PyPI Package
- [X] Documentation of the Code
- [ ] Celebrating :gift_heart:
