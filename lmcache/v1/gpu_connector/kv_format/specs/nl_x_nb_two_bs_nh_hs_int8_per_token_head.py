# SPDX-License-Identifier: Apache-2.0
"""Per-layer NHD num_blocks-first int8 KV: ``NL x [NB, 2, BS, NH, HS+4]``.

A ``list[NL]`` of a 5-D tensor whose leading axis is num_blocks, with the K/V
(size-2) axis following. Produced by vLLM non-MLA flash infer with
``kv_cache_dtype=int8_per_token_head``: the trailing ``4`` elements of each
head hold one fp32 scale, so the physical head stride is ``head_size + 4``.

Structurally identical to ``NL_X_NB_TWO_BS_NH_HS`` — the transfer kernels treat
the trailing axis as opaque bytes, so the offset math matches the base format
with ``hs == head_size + 4``.
"""

# Each spec indexes ``kv_caches`` (Tensor | nested list) per its format, so the
# ``.shape`` / ``[...]`` access is well-defined though mypy cannot prove it.
# mypy: disable-error-code="union-attr,call-overload"
# Standard
from typing import cast

# Third Party
import torch

# First Party
from lmcache.v1.gpu_connector.kv_format.specs.base import KVFormatSpec
import lmcache.c_ops as lmc_ops


class NL_X_NB_TWO_BS_NH_HS_INT8_PER_TOKEN_HEAD_Spec(KVFormatSpec):
    engine_kv_format = lmc_ops.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS_INT8_PER_TOKEN_HEAD
    attention_backends = ("vLLM non-MLA flash infer int8_per_token_head",)
    is_layer_list = True

    def num_layers(self) -> int:
        return len(self.kv_caches)

    def num_blocks(self) -> int:
        return self.kv_caches[0].shape[0]

    def block_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[2]

    def page_buffer_size(self) -> int:
        return self.kv_caches[0].shape[0] * self.kv_caches[0].shape[2]

    def kv_size(self) -> int:
        return 2

    def num_heads(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[3]

    def hidden_dim(self, layer_idx: int = 0) -> int:
        t = self.kv_caches[layer_idx]
        return t.shape[3] * t.shape[4]

    def head_size(self, layer_idx: int = 0) -> int:
        return self.kv_caches[layer_idx].shape[4]

    def tokens_per_layer(self) -> int:
        k = self.kv_caches[0][:, 0].shape
        return k[0] * k[1]

    def elements_per_layer(self) -> int:
        return self.kv_caches[0][:, 0].shape.numel() * 2

    def dtype(self, layer_idx: int = 0) -> torch.dtype:
        return self.kv_caches[layer_idx].dtype

    def data_ptrs(self, layer_indices: list[int]) -> list[int]:
        layers = cast(list[torch.Tensor], self.kv_caches)
        return [layers[i].data_ptr() for i in layer_indices]
