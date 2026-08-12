# SPDX-License-Identifier: Apache-2.0
"""NL_X_NB_TWO_BS_NH_HS_INT8_PER_TOKEN_HEAD: vLLM int8_per_token_head transfers.

The per-layer physical layout is ``[NB, 2, BS, NH, HS+4]`` int8: each head's
trailing 4 elements hold one fp32 scale, and the transfer treats the whole
padded row as opaque bytes (offset math identical to NL_X_NB_TWO_BS_NH_HS).
These tests pin the round-trip against the CPU buffer layout
``[2, NL, T, NH*(HS+4)]``.
"""

# Third Party
import pytest
import torch

if not torch.cuda.is_available():
    pytest.skip("CUDA is not available", allow_module_level=True)

# First Party
import lmcache.c_ops as lmc_ops  # noqa: E402
import lmcache.lmcache_native as lmcache_native  # noqa: E402

if not hasattr(
    lmcache_native.EngineKVFormat, "NL_X_NB_TWO_BS_NH_HS_INT8_PER_TOKEN_HEAD"
):
    pytest.skip(
        "c_ops build lacks NL_X_NB_TWO_BS_NH_HS_INT8_PER_TOKEN_HEAD",
        allow_module_level=True,
    )

_NB = 8  # num blocks
_BS = 16  # tokens per block
_NH = 4  # num kv heads
_HS = 128  # int8 data elements per head
_PAD = 4  # fp32 scale elements per head
_ROW = _HS + _PAD  # padded head stride
_NL = 3  # num layers


def _make_paged() -> tuple[list[torch.Tensor], torch.Tensor]:
    """Per-layer [NB, 2, BS, NH, HS+4] int8 tensors + a device pointer array."""
    caches = [
        torch.randint(-128, 127, (_NB, 2, _BS, _NH, _ROW), dtype=torch.int8).cuda()
        for _ in range(_NL)
    ]
    ptrs = torch.tensor([c.data_ptr() for c in caches], dtype=torch.long).cuda()
    return caches, ptrs


def _write_all(caches, slots, rows):
    for layer in range(_NL):
        for i, slot in enumerate(slots.tolist()):
            b, off = slot // _BS, slot % _BS
            caches[layer][b, :, off, :, :] = rows[layer, i]


def _read_all(caches, slots):
    out = []
    for layer in range(_NL):
        gathered = []
        for slot in slots.tolist():
            b, off = slot // _BS, slot % _BS
            gathered.append(caches[layer][b, :, off].cpu())
        out.append(torch.stack(gathered))
    return torch.stack(out)  # [NL, T, 2, NH, ROW]


def test_int8_pth_roundtrip_value_exact():
    torch.manual_seed(0)
    num_tokens = 37
    page_buffer_size = _NB * _BS
    # Distinct slots: the reference model stores each token at its own slot,
    # so a repeated slot would let one token's write clobber another's read.
    # torch.randperm yields unique slots, so no two tokens share a page slot.
    slots = torch.randperm(page_buffer_size)[:num_tokens].cuda()

    caches, ptrs = _make_paged()

    # Reference rows the CPU buffer should hold after D2H.
    rows = torch.randint(-128, 127, (_NL, num_tokens, 2, _NH, _ROW), dtype=torch.int8)
    _write_all(caches, slots, rows)

    # CPU-side LMCache buffer [2, NL, T, NH*(HS+4)].
    key_value = torch.zeros(
        2, _NL, num_tokens, _NH * _ROW, dtype=torch.int8, device="cpu"
    ).pin_memory()

    # D2H: paged -> LMCache buffer.
    lmc_ops.multi_layer_kv_transfer(
        key_value,
        ptrs,
        slots,
        torch.device("cuda"),
        page_buffer_size,
        lmcache_native.TransferDirection.D2H.value,
        lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS_INT8_PER_TOKEN_HEAD.value,
        block_size=_BS,
    )
    torch.cuda.synchronize()

    # The buffer must hold K and V planes verbatim (data + scale interleaved).
    for layer in range(_NL):
        torch.testing.assert_close(
            key_value[0, layer].view(num_tokens, _NH, _ROW),
            rows[layer, :, 0],
        )
        torch.testing.assert_close(
            key_value[1, layer].view(num_tokens, _NH, _ROW),
            rows[layer, :, 1],
        )

    # H2D: LMCache buffer -> paged; the page must be byte-identical again.
    lmc_ops.multi_layer_kv_transfer(
        key_value,
        ptrs,
        slots,
        torch.device("cuda"),
        page_buffer_size,
        lmcache_native.TransferDirection.H2D.value,
        lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS_INT8_PER_TOKEN_HEAD.value,
        block_size=_BS,
    )
    torch.cuda.synchronize()

    torch.testing.assert_close(_read_all(caches, slots), rows)


def test_int8_pth_h2d_alone():
    """H2D alone (LMCache buffer -> paged) must land the padded rows verbatim."""
    torch.manual_seed(2)
    num_tokens = 24
    page_buffer_size = _NB * _BS
    slots = torch.randperm(page_buffer_size)[:num_tokens].cuda()
    # Keep the edge slots exercised too (block boundaries, tail of block).
    slots[:5] = torch.tensor([0, _BS - 1, _BS, _BS + 1, page_buffer_size - 1])

    caches, ptrs = _make_paged()
    rows = torch.randint(-128, 127, (_NL, num_tokens, 2, _NH, _ROW), dtype=torch.int8)

    key_value = torch.zeros(
        2, _NL, num_tokens, _NH * _ROW, dtype=torch.int8, device="cpu"
    ).pin_memory()
    key_value.copy_(rows.permute(2, 0, 1, 3, 4).reshape(2, _NL, num_tokens, -1))

    lmc_ops.multi_layer_kv_transfer(
        key_value,
        ptrs,
        slots,
        torch.device("cuda"),
        page_buffer_size,
        lmcache_native.TransferDirection.H2D.value,
        lmcache_native.EngineKVFormat.NL_X_NB_TWO_BS_NH_HS_INT8_PER_TOKEN_HEAD.value,
        block_size=_BS,
    )
    torch.cuda.synchronize()
    torch.testing.assert_close(_read_all(caches, slots), rows)
