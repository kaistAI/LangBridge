import torch
from torch import einsum
import torch.nn as nn
from einops import rearrange, repeat
from einops_exts import rearrange_many


class Linear(nn.Module):
    """
    Linear layer with optional bias and activation function.
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=True)

    def forward(self, x):
        x = self.linear(x)
        return x


class LinearWithAddedEos(nn.Module):
    """
    Linear layer with optional bias and activation function.
    """

    def __init__(self, dim, out_dim):
        super().__init__()
        self.linear = nn.Linear(dim, out_dim, bias=True)
        self.enc_eos = nn.Parameter(torch.randn(out_dim))

    def forward(self, x, *args):
        x = self.linear(x)
        eos = repeat(self.enc_eos, 'd -> b d', b=x.shape[0])
        eos = rearrange(eos, 'b d -> b 1 d')
        x = torch.cat((x, eos), dim=1)
        return x


class FFNWithAddedEos(nn.Module):
    def __init__(self, dim, out_dim):
        super().__init__()
        self.ffn = FeedForward(dim, out_dim)
        self.enc_eos = nn.Parameter(torch.randn(out_dim))

    def forward(self, x, *args):
        x = self.ffn(x)
        eos = repeat(self.enc_eos, 'd -> b d', b=x.shape[0])
        eos = rearrange(eos, 'b d -> b 1 d')
        x = torch.cat((x, eos), dim=1)
        return x


def FeedForward(dim, out_dim, mult=2, act='gelu'):
    """
    lucidrains implementation, slightly modified with the act parameter.
    """

    acts = dict(
        gelu=nn.GELU,
        relu=nn.ReLU
    )

    assert act in acts, f"act. can only be one of {acts.keys()}"

    inner_dim = int(dim * mult)
    return nn.Sequential(
        nn.Linear(dim, inner_dim),
        acts[act](),
        nn.Linear(inner_dim, out_dim),
    )


class PerceiverAttentionLayer(nn.Module):
    def __init__(
            self,
            *,
            dim,
            dim_head=64,
            heads=8
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        self.dim_head = dim_head
        inner_dim = dim_head * heads

        # trainable components of PerceiverAttentionLayer
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

    def forward(self, features, masks, latents):
        """
        Latent vectors are cross-attending to the visual features x.
        :param x:       Tensor (n_batch, n_features or seq_len, dim)
                        visual features
        :param latents: Tensor (n_batch, n_latents, dim)
                        latent learnt vectors from which the queries are computed.
                        Actually the same, just replicated in n_batch and n_frames dimension.
        :return:        Tensor (n_batch, n_latents, dim)
        """
        assert features.ndim == 3
        assert latents.ndim == 3
        assert features.shape[0] == latents.shape[0]
        assert features.shape[2] == latents.shape[2]
        assert masks.shape == features.shape[:2]

        n_heads = self.heads
        n_batch, n_features, dim = features.shape
        n_queries = latents.shape[1]

        # layer normalization, as usual
        x = self.norm_media(features)
        latents = self.norm_latents(latents)

        # queries
        # compute the queries from the latents, for all attention heads simultaneously.
        q = self.to_q(latents)
        q = rearrange(q, 'b q (h d) -> b h q d', h=n_heads)
        assert q.shape == torch.Size(
            [n_batch, n_heads, n_queries, self.dim_head])

        kv_input = torch.cat((x, latents), dim=-2)
        n_features_latents = n_features + n_queries

        kv_mask = torch.cat(
            (masks, torch.ones(n_batch, n_queries).to(masks.device)), dim=-1)  # (b, n_features + n_queries)
        kv_mask = rearrange(kv_mask, 'b n -> b 1 1 n')
        assert kv_mask.shape == torch.Size([n_batch, 1, 1, n_features_latents])

        # keys, values
        k = self.to_k(kv_input)
        v = self.to_v(kv_input)
        # batch, features, (heads, dim)

        k, v = rearrange_many((k, v), 'b f (h d) -> b h f d', h=n_heads)
        assert v.shape == torch.Size(
            [n_batch, n_heads, n_features_latents, self.dim_head])

        # scale queries?
        q = q * self.scale

        # attention
        # attention scores
        # sim = einsum('... i d, ... j d  -> ... i j', q, k)
        sim = einsum('b h q d, b h f d -> b h q f', q, k)

        # Is this for numerical stability? Does not affect the result of the softmax operation
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()

        # mask out the padded values
        sim = sim.masked_fill(~kv_mask.bool(), float('-inf'))

        alphas = sim.softmax(dim=-1)

        out = einsum('b h q f, b h f v -> b h q v', alphas, v)

        out = rearrange(out, 'b h q v -> b q (h v)')  # merge heads
        return self.to_out(out)


class PerceiverResampler(nn.Module):
    def __init__(
        self,
        *,
        dim,
        out_dim,
        depth=1,
        dim_head=64,
        heads=8,
        num_latents=64,
        ff_mult=4,
        act='gelu'
    ):
        """
        :param dim:             length of the visual features and of the queries (-> thus also length of the keys)
        :param depth:           number of attention layers
        :param dim_head:        inner dimensionality of the q, k, v vectors per attention head
        :param heads:           number of attention heads
        :param num_latents:     number of queries, default 64 as in the flamingo paper
        :param ff_mult:         factor for the number of inner neurons in the feedforward layer
        """
        super().__init__()

        self.dim = dim
        self.out_dim = out_dim
        self.n_queries = num_latents

        # latents are not the queries themselves, but latent vectors from which the queries are computed.
        # (same dimension as the length of the features
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.enc_eos = nn.Parameter(torch.randn(out_dim))
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PerceiverAttentionLayer(
                    dim=dim, dim_head=dim_head, heads=heads),
                FeedForward(
                    dim=dim, out_dim=dim, mult=ff_mult, act=act)
            ]))
        self.out_proj = nn.Linear(dim, out_dim, bias=False)
        self.dropout = nn.Dropout(0.1)

        # layer normalization takes as input the query vector length
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x_f, x_mask):
        assert x_f.ndim == 3  # (b s d)
        assert x_mask.ndim == 2  # (b s)

        n_batches = x_f.shape[0]
        dim = x_f.shape[2]

        assert dim == self.dim

        x = repeat(self.latents, 'q d -> b q d', b=n_batches)

        for attn, ffw in self.layers:  # type: ignore
            x = x + attn(x_f, x_mask, x)
            x = x + ffw(x)
            x = self.dropout(x)
        assert x.shape == torch.Size([n_batches, self.n_queries, self.dim])

        x = self.out_proj(x)
        eos = repeat(self.enc_eos, 'd -> b d', b=n_batches)
        eos = rearrange(eos, 'b d -> b 1 d')
        x = torch.cat((x, eos), dim=1)

        norm = self.norm(x)
        return norm
