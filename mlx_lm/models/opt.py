# Copyright © 2025 Apple Inc.

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
    ffn_dim: int = 3072
    max_position_embeddings: int = 2048
    num_attention_heads: int = 12
    word_embed_proj_dim: int = None
    layer_norm_elementwise_affine: bool = True
    _remove_final_layer_norm: bool = False
    do_layer_norm_before: bool = True
    enable_bias: bool = True

    def __post_init__(self):
        if self.word_embed_proj_dim is None:
            self.word_embed_proj_dim = self.hidden_size


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.num_heads = args.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.scale = self.head_dim**-0.5

        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=args.enable_bias)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=args.enable_bias)
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=args.enable_bias)
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=args.enable_bias)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries = self.q_proj(x)
        keys = self.k_proj(x)
        values = self.v_proj(x)

        # Prepare the queries, keys and values for the attention computation
        queries = queries.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.num_heads, -1).transpose(0, 2, 1, 3)

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

        self.fc1 = nn.Linear(args.hidden_size, args.ffn_dim, bias=args.enable_bias)
        self.fc2 = nn.Linear(args.ffn_dim, args.hidden_size, bias=args.enable_bias)

    def __call__(self, x) -> mx.array:
        return self.fc2(nn.relu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.do_layer_norm_before = args.do_layer_norm_before
        self.self_attn = Attention(args)
        self.self_attn_layer_norm = nn.LayerNorm(
            args.hidden_size,
            eps=1e-5,
            affine=args.layer_norm_elementwise_affine,
        )
        self.mlp = MLP(args)
        self.final_layer_norm = nn.LayerNorm(
            args.hidden_size,
            eps=1e-5,
            affine=args.layer_norm_elementwise_affine,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = x

        if self.do_layer_norm_before:
            x = self.self_attn_layer_norm(x)

        x = self.self_attn(x, mask, cache)
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

        self.embed_tokens = nn.Embedding(args.vocab_size, args.word_embed_proj_dim)
        self.embed_positions = nn.Embedding(args.max_position_embeddings, args.hidden_size)

        if args.word_embed_proj_dim != args.hidden_size:
            self.project_in = nn.Linear(args.word_embed_proj_dim, args.hidden_size, bias=False)
        else:
            self.project_in = None

        if args.word_embed_proj_dim != args.hidden_size:
            self.project_out = nn.Linear(args.hidden_size, args.word_embed_proj_dim, bias=False)
        else:
            self.project_out = None

        self.layers = [TransformerBlock(args) for _ in range(args.num_hidden_layers)]

        if args.do_layer_norm_before and not args._remove_final_layer_norm:
            self.final_layer_norm = nn.LayerNorm(
                args.hidden_size,
                eps=1e-5,
                affine=args.layer_norm_elementwise_affine,
            )
        else:
            self.final_layer_norm = None

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        B, L = inputs.shape

        hidden_states = self.embed_tokens(inputs)

        if self.project_in is not None:
            hidden_states = self.project_in(hidden_states)

        if cache is None:
            cache = [None] * len(self.layers)

        offset = 0
        if cache[0] is not None:
            offset = cache[0].offset

        # OPT uses offset of 2 for position embeddings
        position_ids = mx.arange(L) + offset + 2
        position_embeddings = self.embed_positions(position_ids)
        hidden_states = hidden_states + position_embeddings

        mask = create_attention_mask(hidden_states, cache[0])

        for layer, c in zip(self.layers, cache):
            hidden_states = layer(hidden_states, mask, cache=c)

        if self.final_layer_norm is not None:
            hidden_states = self.final_layer_norm(hidden_states)

        if self.project_out is not None:
            hidden_states = self.project_out(hidden_states)

        return hidden_states


class OPTModel(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.decoder = OPTDecoder(args)

    def __call__(
        self,
        inputs: mx.array,
        cache=None,
    ):
        return self.decoder(inputs, cache)


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
        # OPT weight names in HuggingFace format need to be mapped to MLX format
        new_weights = {}
        for key, value in weights.items():
            # Remove "model." prefix if present for lm_head
            if key == "lm_head.weight":
                new_weights[key] = value
            else:
                new_weights[key] = value
        return new_weights

    @property
    def layers(self):
        return self.model.decoder.layers
