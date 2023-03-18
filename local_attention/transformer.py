from tensorflow.keras import layers

class LocalMHA(layers.Layer):
    def __init__(
        self,
        window_size,
        dim_head = 64,
        heads = 8,
        dropout = 0.,
        causal = False,
        prenorm = False,
        qk_rmsnorm = False,
        qk_scale = 8,
        # use_xpos = False,
        xpos_scale_base = None,
        **kwargs
    ):
        super().__init__()        
        self.inner_dim = dim_head * heads
        self.window_size = window_size
        self.causal = causal

        self.heads = heads
        self.qk_rmsnorm = qk_rmsnorm
        self.prenorm = prenorm

    def build(self, input_shape):
        dim = input_shape[-1]
        self.norm = layers.LayerNormalization(axis=-1) if self.prenorm else None

        self.to_qkv = layers.Dense(self.inner_dim * 3, use_bias = False)


        # if self.qk_rmsnorm:
        #     self.q_scale = nn.Parameter(torch.ones(dim_head))
        #     self.k_scale = nn.Parameter(torch.ones(dim_head))

        self.attn_fn = LocalAttention(
            window_size = self.window_size,
            causal = self.causal,
            autopad = True,
            scale = (self.qk_scale if self.qk_rmsnorm else None),
            exact_windowsize = True,
            # use_xpos = self.use_xpos,
            # xpos_scale_base = self.xpos_scale_base,
            # **kwargs
        )

        self.to_out = layers.Dense(dim, use_bias = False)

    def call(self, x, mask = None, attn_bias = None):
        if exists(self.norm):
            x = self.norm(x)

        q, k, v = tf.split(self.to_qkv(x),3, axis=-1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), (q, k, v)) 

        # if self.qk_rmsnorm:
        #     q, k = map(l2norm, (q, k))
        #     q = q * self.q_scale
        #     k = k * self.k_scale

        out = self.attn_fn(q, k, v, mask = mask)

        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)
