# Copyright © 2025 Apple Inc.

import math
from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "bloom"
    vocab_size: int = 250880
    hidden_size: int = 1024
    n_layer: int = 24
    n_head: int = 16
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02
    apply_residual_connection_post_layernorm: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    pretraining_tp: int = 1
    slow_but_exact: bool = False

    # Derived parameters
    @property
    def num_hidden_layers(self):
        return self.n_layer

    @property
    def num_attention_heads(self):
        return self.n_head


def build_alibi_slopes(num_heads: int) -> mx.array:
    """Build ALiBi slopes for attention bias."""

    def get_slopes_power_of_2(n):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * (ratio**i) for i in range(n)]

    if math.log2(num_heads).is_integer():
        slopes = get_slopes_power_of_2(num_heads)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))
        slopes = get_slopes_power_of_2(closest_power_of_2)
        slopes_2 = get_slopes_power_of_2(2 * closest_power_of_2)
        slopes_2 = slopes_2[0::2][: num_heads - closest_power_of_2]
        slopes = slopes + slopes_2

    return mx.array(slopes)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.num_heads = args.n_head
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        # BLOOM uses a fused query/key/value projection
        self.query_key_value = nn.Linear(self.hidden_size, 3 * self.hidden_size, bias=True)
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

        # ALiBi slopes
        self.alibi_slopes = build_alibi_slopes(self.num_heads)

    def _build_alibi_bias(self, seq_len: int, offset: int = 0) -> mx.array:
        """Build ALiBi attention bias."""
        # Create position indices
        positions = mx.arange(seq_len) + offset
        # Create relative positions matrix
        # Shape: [seq_len, seq_len + offset] for causal attention
        if offset > 0:
            key_positions = mx.arange(seq_len + offset)
        else:
            key_positions = positions
        relative_positions = positions[:, None] - key_positions[None, :]
        # Apply slopes: [num_heads, 1, 1] * [1, seq_len, key_len]
        alibi = self.alibi_slopes[:, None, None] * relative_positions[None, :, :]
        return alibi

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        qkv = self.query_key_value(x)
        qkv = qkv.reshape(B, L, self.num_heads, 3, self.head_dim)
        queries = qkv[:, :, :, 0, :].transpose(0, 2, 1, 3)
        keys = qkv[:, :, :, 1, :].transpose(0, 2, 1, 3)
        values = qkv[:, :, :, 2, :].transpose(0, 2, 1, 3)

        offset = 0
        if cache is not None:
            offset = cache.offset
            keys, values = cache.update_and_fetch(keys, values)

        # Build ALiBi bias
        alibi = self._build_alibi_bias(L, offset)

        # Combine with causal mask
        if mask is not None:
            # mask shape: [B, 1, L, key_len] or [B, heads, L, key_len]
            # alibi shape: [heads, L, key_len]
            # Expand alibi to match batch dimension
            alibi = alibi[None, :, :, :]  # [1, heads, L, key_len]
            # Add alibi to mask (mask has -inf for masked positions)
            combined_mask = mask + alibi
        else:
            combined_mask = alibi[None, :, :, :]

        # Standard attention computation
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * self.scale
        scores = scores + combined_mask
        weights = mx.softmax(scores, axis=-1)
        output = weights @ values

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.dense_h_to_4h = nn.Linear(args.hidden_size, 4 * args.hidden_size, bias=True)
        self.dense_4h_to_h = nn.Linear(4 * args.hidden_size, args.hidden_size, bias=True)

    def __call__(self, x) -> mx.array:
        x = self.dense_h_to_4h(x)
        x = nn.gelu_approx(x)
        return self.dense_4h_to_h(x)


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.apply_residual_connection_post_layernorm = args.apply_residual_connection_post_layernorm

        self.input_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.self_attention = Attention(args)
        self.post_attention_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.mlp = MLP(args)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # Pre-LayerNorm
        residual = x
        x = self.input_layernorm(x)

        if self.apply_residual_connection_post_layernorm:
            residual = x

        x = self.self_attention(x, mask, cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)

        if self.apply_residual_connection_post_layernorm:
            residual = x

        x = self.mlp(x)
        x = residual + x

        return x


class BloomModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        self.word_embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.word_embeddings_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)

        self.h = [TransformerBlock(args) for _ in range(args.n_layer)]

        self.ln_f = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        x = self.word_embeddings(inputs)
        x = self.word_embeddings_layernorm(x)

        if cache is None:
            cache = [None] * len(self.h)

        mask = create_attention_mask(x, cache[0])

        for layer, c in zip(self.h, cache):
            x = layer(x, mask, cache=c)

        return self.ln_f(x)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = BloomModel(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.transformer(inputs, cache)
        # Tie word embeddings
        out = self.transformer.word_embeddings.as_linear(out)
        return out

    def sanitize(self, weights):
        new_weights = {}
        for key, value in weights.items():
            # Map HuggingFace weight names to MLX format
            new_key = key
            if key.startswith("transformer."):
                new_key = key.replace("transformer.", "")
            new_weights[new_key] = value
        return new_weights

    @property
    def layers(self):
        return self.transformer.h
