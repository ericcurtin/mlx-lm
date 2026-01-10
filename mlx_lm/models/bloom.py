# Copyright © 2025 Apple Inc.
# BLOOM model implementation with ALiBi positional encoding

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_causal_mask


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "bloom"
    vocab_size: int = 250880
    hidden_size: int = 1024
    n_layer: int = 24
    n_head: int = 16
    layer_norm_epsilon: float = 1e-5
    # BLOOM doesn't use num_hidden_layers in config, uses n_layer instead
    num_hidden_layers: int = None

    def __post_init__(self):
        if self.num_hidden_layers is None:
            self.num_hidden_layers = self.n_layer


def build_alibi_slopes(num_heads: int) -> mx.array:
    """
    Build ALiBi slopes for the given number of attention heads.

    ALiBi uses powers of 2^(-8/n) for slopes, where n is the number of heads.
    For heads that don't fit a power of 2, we interpolate.
    """
    import math

    closest_power_of_2 = 2 ** math.floor(math.log2(num_heads))

    base = 2 ** (-(2 ** -(math.log2(closest_power_of_2) - 3)))
    powers = mx.arange(1, closest_power_of_2 + 1)
    slopes = mx.array(base) ** powers

    if closest_power_of_2 != num_heads:
        extra_base = 2 ** (-(2 ** -(math.log2(2 * closest_power_of_2) - 3)))
        num_remaining_heads = min(closest_power_of_2, num_heads - closest_power_of_2)
        extra_powers = mx.arange(1, 2 * num_remaining_heads + 1, 2)
        extra_slopes = mx.array(extra_base) ** extra_powers
        slopes = mx.concatenate([slopes, extra_slopes], axis=0)

    return slopes[:num_heads]


def compute_alibi_bias(seq_len: int, num_heads: int) -> mx.array:
    """
    Compute ALiBi attention bias for the given sequence length.

    Returns a tensor of shape [1, num_heads, seq_len, seq_len]
    """
    slopes = build_alibi_slopes(num_heads)

    # Create position differences matrix
    # position_ids: [seq_len]
    position_ids = mx.arange(seq_len)
    # relative_positions: [seq_len, seq_len]
    # For position i attending to position j, bias = slope * (j - i)
    # But we only allow attending to positions <= current position (causal)
    relative_positions = position_ids[None, :] - position_ids[:, None]

    # Shape: [1, num_heads, seq_len, seq_len]
    alibi_bias = slopes[None, :, None, None] * relative_positions[None, None, :, :]

    return alibi_bias.astype(mx.float32)


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.n_head
        self.head_dim = self.hidden_size // self.num_heads

        # BLOOM uses fused QKV projection
        self.query_key_value = nn.Linear(
            self.hidden_size, 3 * self.hidden_size, bias=True
        )
        self.dense = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        alibi: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        # Fused QKV projection
        qkv = self.query_key_value(x)

        # Split into Q, K, V
        qkv = qkv.reshape(B, L, self.num_heads, 3 * self.head_dim)
        queries, keys, values = mx.split(qkv, 3, axis=-1)

        # Transpose to [B, num_heads, L, head_dim]
        queries = queries.transpose(0, 2, 1, 3)
        keys = keys.transpose(0, 2, 1, 3)
        values = values.transpose(0, 2, 1, 3)

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        kv_len = keys.shape[2]

        # Compute attention scores
        scale = self.head_dim**-0.5
        scores = (queries @ keys.transpose(0, 1, 3, 2)) * scale

        # Add ALiBi bias
        # Need to slice alibi to match current key length
        if alibi.shape[-1] < kv_len:
            # Recompute alibi for longer sequence
            alibi = compute_alibi_bias(kv_len, self.num_heads)

        # For incremental decoding, we need to adjust the alibi slice
        q_len = queries.shape[2]
        alibi_slice = alibi[:, :, -q_len:, :kv_len]
        scores = scores + alibi_slice

        # Apply causal mask
        if mask is not None:
            if isinstance(mask, str) and mask == "causal":
                # Create causal mask
                mask = create_causal_mask(q_len, kv_len - q_len)
            if mask.dtype == mx.bool_:
                scores = mx.where(mask, scores, mx.finfo(scores.dtype).min)
            else:
                scores = scores + mask

        # Softmax and output
        weights = mx.softmax(scores, axis=-1)
        output = weights @ values

        # Reshape back
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.dense(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        # BLOOM uses 4x hidden size for intermediate
        self.dense_h_to_4h = nn.Linear(self.hidden_size, 4 * self.hidden_size, bias=True)
        self.dense_4h_to_h = nn.Linear(4 * self.hidden_size, self.hidden_size, bias=True)

    def __call__(self, x: mx.array) -> mx.array:
        x = self.dense_h_to_4h(x)
        x = nn.gelu(x)
        x = self.dense_4h_to_h(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.input_layernorm = nn.LayerNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.self_attention = Attention(args)
        self.post_attention_layernorm = nn.LayerNorm(
            args.hidden_size, eps=args.layer_norm_epsilon
        )
        self.mlp = MLP(args)

    def __call__(
        self,
        x: mx.array,
        alibi: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        # Pre-LayerNorm architecture
        residual = x
        x = self.input_layernorm(x)
        x = self.self_attention(x, alibi=alibi, mask=mask, cache=cache)
        x = residual + x

        residual = x
        x = self.post_attention_layernorm(x)
        x = self.mlp(x)
        x = residual + x

        return x


class BloomModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.embed_dim = args.hidden_size
        self.num_heads = args.n_head

        self.word_embeddings = nn.Embedding(args.vocab_size, self.embed_dim)
        self.word_embeddings_layernorm = nn.LayerNorm(
            self.embed_dim, eps=args.layer_norm_epsilon
        )
        self.h = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]
        self.ln_f = nn.LayerNorm(self.embed_dim, eps=args.layer_norm_epsilon)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        B, L = inputs.shape

        # Get token embeddings
        x = self.word_embeddings(inputs)
        x = self.word_embeddings_layernorm(x)

        if cache is None:
            cache = [None] * len(self.h)

        # Compute ALiBi bias
        # For caching, we need to compute for the full sequence length
        total_len = L
        if cache[0] is not None:
            total_len = cache[0].offset + L

        alibi = compute_alibi_bias(total_len, self.num_heads)

        # Create causal mask
        mask = "causal" if L > 1 else None

        for layer, c in zip(self.h, cache):
            x = layer(x, alibi=alibi, mask=mask, cache=c)

        x = self.ln_f(x)
        return x


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.transformer = BloomModel(args)
        # BLOOM ties input and output embeddings
        self.lm_head = None

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.transformer(inputs, cache)
        # Use tied embeddings for language model head
        return self.transformer.word_embeddings.as_linear(out)

    def sanitize(self, weights):
        new_weights = {}

        for key, value in weights.items():
            new_key = key

            # Handle transformer prefix
            if key.startswith("transformer."):
                new_key = key.replace("transformer.", "transformer.", 1)
            elif not key.startswith("transformer.") and not key.startswith("lm_head"):
                new_key = f"transformer.{key}"

            # Skip lm_head if embeddings are tied (BLOOM ties them)
            if "lm_head" in key:
                continue

            new_weights[new_key] = value

        return new_weights

    @property
    def layers(self):
        return self.transformer.h
