# https://github.com/sayakpaul/maxim-tf/blob/main/benchmark_xla.py

from local_attention_tf.text import TextClassifier
import timeit
import numpy as np

vocab_size = 20000  
maxlen = 200 


classifier = TextClassifier(maxlen, vocab_size, 2)


dummy_inputs = tf.random.uniform((1, 200))


def benchmark_regular_model():
    # Warmup
    print("Benchmarking TF model...")
    for _ in range(2):
        _ = classifier(dummy_inputs)

    # Timing
    tf_runtimes = timeit.repeat(
        lambda: classifier(dummy_inputs, training=False), number=1, repeat=10
    )
    print(f"Average latency (seconds): {np.mean(tf_runtimes)}.")




def benchmark_xla_model():
    # Warmup
    print("Benchmarking Jit-compiled TF model...")
    for _ in range(2):
        _ = get_xla_compiled()

    # Timing
    tf_runtimes = timeit.repeat(lambda: get_xla_compiled(), number=1, repeat=10)
    print(f"Average latency (seconds): {np.mean(tf_runtimes)}.")
if __name__ == '__main__':
    benchmark_regular_model()
    benchmark_xla_model()
    
