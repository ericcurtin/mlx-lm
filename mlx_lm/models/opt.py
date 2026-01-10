# Copyright © 2025 Apple Inc.
# OPT (Open Pre-trained Transformer) model implementation

from dataclasses import dataclass
from typing import Any, Optional

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str = "opt"
    vocab_size: int = 50272
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 2048
    layer_norm_elementwise_affine: bool = True
    word_embed_proj_dim: int = None
    ffn_dim: int = None
    activation_function: str = "relu"
    do_layer_norm_before: bool = True
    # OPT uses an offset of 2 for learned positional embeddings
    pad_token_id: int = 1

    def __post_init__(self):
        if self.word_embed_proj_dim is None:
            self.word_embed_proj_dim = self.hidden_size
        if self.ffn_dim is None:
            self.ffn_dim = self.intermediate_size


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=True)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, _ = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Reshape to [B, num_heads, L, head_dim]
        queries = queries.reshape(B, L, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )
        keys = keys.reshape(B, L, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_heads, self.head_dim).transpose(
            0, 2, 1, 3
        )

        if cache is not None:
            keys, values = cache.update_and_fetch(keys, values)

        output = scaled_dot_product_attention(
            queries, keys, values, cache=cache, scale=self.scale, mask=mask
        )

        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.out_proj(output)


class MLP(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.fc1 = nn.Linear(args.hidden_size, args.ffn_dim, bias=True)
        self.fc2 = nn.Linear(args.ffn_dim, args.hidden_size, bias=True)
        self.activation_fn = args.activation_function

    def __call__(self, x: mx.array) -> mx.array:
        x = self.fc1(x)
        if self.activation_fn == "relu":
            x = nn.relu(x)
        else:
            x = nn.gelu(x)
        x = self.fc2(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.do_layer_norm_before = args.do_layer_norm_before
        self.self_attn = Attention(args)
        self.self_attn_layer_norm = nn.LayerNorm(args.hidden_size)
        self.mlp = MLP(args)
        self.final_layer_norm = nn.LayerNorm(args.hidden_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x

        if self.do_layer_norm_before:
            x = self.self_attn_layer_norm(x)

        x = self.self_attn(x, mask=mask, cache=cache)
        x = residual + x

        if not self.do_layer_norm_before:
            x = self.self_attn_layer_norm(x)

        residual = x

        if self.do_layer_norm_before:
            x = self.final_layer_norm(x)

        x = self.mlp(x)
        x = residual + x

        if not self.do_layer_norm_before:
            x = self.final_layer_norm(x)

        return x


class OPTDecoder(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args

        # OPT uses a learned position embedding with offset=2
        # The embedding has max_position_embeddings + 2 positions
        self.embed_tokens = nn.Embedding(args.vocab_size, args.word_embed_proj_dim)
        self.embed_positions = nn.Embedding(
            args.max_position_embeddings, args.hidden_size
        )

        # Optional projection if word_embed_proj_dim != hidden_size
        if args.word_embed_proj_dim != args.hidden_size:
            self.project_in = nn.Linear(
                args.word_embed_proj_dim, args.hidden_size, bias=False
            )
            self.project_out = nn.Linear(
                args.hidden_size, args.word_embed_proj_dim, bias=False
            )
        else:
            self.project_in = None
            self.project_out = None

        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]

        # OPT applies final layer norm if do_layer_norm_before is True
        if args.do_layer_norm_before:
            self.final_layer_norm = nn.LayerNorm(args.hidden_size)
        else:
            self.final_layer_norm = None

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        _, L = inputs.shape

        # Get token embeddings
        x = self.embed_tokens(inputs)

        if self.project_in is not None:
            x = self.project_in(x)

        # Get position ids (OPT uses offset of 2)
        if cache is None:
            cache = [None] * len(self.layers)

        offset = 0
        if cache[0] is not None:
            offset = cache[0].offset

        # OPT uses attention_mask to calculate position_ids
        # Position ids start from 2 (offset)
        position_ids = mx.arange(L) + offset + 2

        # Add position embeddings
        x = x + self.embed_positions(position_ids)

        mask = create_attention_mask(x, cache[0])

        for layer, c in zip(self.layers, cache):
            x = layer(x, mask=mask, cache=c)

        if self.final_layer_norm is not None:
            x = self.final_layer_norm(x)

        if self.project_out is not None:
            x = self.project_out(x)

        return x


class OPTModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.decoder = OPTDecoder(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        return self.decoder(inputs, cache=cache)


class Model(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = OPTModel(args)
        self.lm_head = nn.Linear(args.word_embed_proj_dim, args.vocab_size, bias=False)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        out = self.model(inputs, cache)
        return self.lm_head(out)

    def sanitize(self, weights):
        # Map HuggingFace weight names to our weight names
        new_weights = {}

        for key, value in weights.items():
            # Remove "model." prefix if present
            new_key = key

            # Handle lm_head
            if key == "lm_head.weight":
                new_weights[key] = value
                continue

            # Add model. prefix for decoder weights
            if not key.startswith("model."):
                new_key = f"model.{key}"

            new_weights[new_key] = value

        return new_weights

    @property
    def layers(self):
        return self.model.decoder.layers
