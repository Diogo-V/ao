import numpy
import torch
from typing import Tuple

from torchao.sparsity.aaa import (
    mask_creator, 
    sparse_semi_structured_from_dense_cutlass,
    sparse_semi_structured_to_dense_cutlass
)
from torchao.sparsity.bbb import (
    marlin_24_perm, 
    marlin_24_scale_perm, 
    marlin_24_scale_perm_single
)
from torchao.sparsity.ccc import (
    get_pack_factor, quantize_weights
)

MARLIN_TILE = 16


def pack_to_marlin_24(
        q_w_24: torch.Tensor, 
        scales: torch.Tensor, 
        num_bits: int, 
        group_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    in_features, out_features = q_w_24.shape

    # Compress quantized weight
    q_w_24_comp, meta = compress_quantized_24_weight(
        q_w_24, in_features, out_features, num_bits
    )

    in_features_comp = in_features // 2
    # assert decompress_quantized_24_weight(q_w_24_comp, meta, in_features_comp, out_features, num_bits) == q_w_24

    # Reformat to marlin
    marlin_24_q_w_comp = to_marlin_weights(
        q_w_24_comp, in_features_comp, out_features,
        num_bits, marlin_24_perm[num_bits]
    )

    marlin_24_s = to_marlin_scales(
        scales, in_features, out_features, group_size,
        marlin_24_scale_perm[num_bits],
        marlin_24_scale_perm_single[num_bits]
    )

    return marlin_24_q_w_comp, marlin_24_s, meta


def unpack_from_marlin_24(
        q_w_24_comp: torch.Tensor, 
        scales: torch.Tensor, 
        meta: torch.Tensor, 
        tiles: int, 
        original_shape: torch.Size,
        group_size: int,
        num_bits: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:

    # Unpacks the scales 
    unpacked_scales = from_marlin_scale(
        scales, *original_shape, group_size,
        marlin_24_scale_perm[num_bits],
        marlin_24_scale_perm_single[num_bits]
    )


def from_marlin_weights(q_w_24, size_k, size_n, perm) -> torch.Tensor:
    # TODO(diogo): WIP
    pass


def from_marlin_scale(s, size_k, size_n, group_size, scale_perm, scale_perm_single) -> torch.Tensor:
    s = s.reshape((-1, size_n)).contiguous()

    if group_size < size_k and group_size != -1:
        reverse_scale_perms = torch.tensor(scale_perm).argsort()
        s = s.reshape((-1, len(scale_perm)))[:, reverse_scale_perms]
    else:
        reverse_scale_perms = torch.tensor(scale_perm_single).argsort()
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]

    return s.reshape(-1).contiguous()


# TODO(diogo): WIP
def decompress_quantized_24_weight(q_24_comp, meta, size_k, size_n, num_bits) -> torch.Tensor:
    assert q_24_comp.shape == (size_k, size_n)

    # Resize meta back to its original shape
    meta = meta.resize_(meta.shape[0] * 2, meta.shape[1] // 2)

    # Remove zp to normalize over 0
    max_q_val = (1 << num_bits) - 1
    zp = (max_q_val + 1) // 2
    q_24_no_zp_comp = q_24_comp - zp

    # Decompress
    q_24_no_zp_comp = q_24_no_zp_comp.t().contiguous()
    q_24_no_zp = sparse_semi_structured_to_dense_cutlass(q_24_no_zp_comp, meta)
    q_24_no_zp = q_24_no_zp.t().contiguous()

    # Restore zp
    q_24 = q_24_no_zp + zp

    return q_24


def marlin_permute_weights(q_w, size_k, size_n, perm, tile=MARLIN_TILE):
    assert q_w.shape == (size_k, size_n)
    assert size_k % tile == 0, f"size_k = {size_k}, tile = {tile}"
    assert size_n % tile == 0, f"size_k = {size_n}, tile = {tile}"

    # Permute weights to 16x64 marlin tiles
    q_w = q_w.reshape((size_k // tile, tile, size_n // tile, tile))
    q_w = q_w.permute((0, 2, 1, 3))
    q_w = q_w.reshape((size_k // tile, size_n * tile))

    q_w = q_w.reshape((-1, perm.numel()))[:, perm].reshape(q_w.shape)

    return q_w


def to_marlin_weights(q_w, size_k, size_n, num_bits, perm):
    # Permute
    q_w = marlin_permute_weights(q_w, size_k, size_n, perm)

    # Pack
    pack_factor = get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(numpy.uint32)

    q_packed = numpy.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=numpy.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(numpy.int32)).to(orig_device)

    return q_packed


def to_marlin_scales(s, size_k, size_n, group_size, scale_perm, scale_perm_single):
    if group_size < size_k and group_size != -1:
        s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    else:
        s = s.reshape((-1, len(scale_perm_single)))[:, scale_perm_single]
    s = s.reshape((-1, size_n)).contiguous()
    return s


def inject_24(w, size_k, size_n):
    assert w.shape == (size_k, size_n)
    mask = mask_creator(w.t()).t().cuda().bool()
    return (mask * w).contiguous(), mask.contiguous()


def compress_quantized_24_weight(q_24, size_k, size_n, num_bits):
    assert q_24.shape == (size_k, size_n)

    # Remove zp to normalize over 0
    max_q_val = (1 << num_bits) - 1
    zp = (max_q_val + 1) // 2
    q_24_no_zp = q_24 - zp

    # Compress
    q_24_no_zp = q_24_no_zp.t().contiguous()
    q_24_no_zp_comp, meta = sparse_semi_structured_from_dense_cutlass(q_24_no_zp)
    q_24_no_zp_comp = q_24_no_zp_comp.t().contiguous()

    # Restore zp
    q_24_comp = q_24_no_zp_comp + zp

    # Resize meta to its actual shape (without moving any data)
    meta = meta.resize_(meta.shape[1] // 2, meta.shape[0] * 2)

    return q_24_comp, meta


def marlin_24_quantize(
    w: torch.Tensor,
    num_bits: int,
    group_size: int,
):
    size_k, size_n = w.shape

    # Normalize group_size
    if group_size == -1:
        group_size = size_k
    assert group_size <= size_k

    # Inject 2:4 sparsity
    w_24, mask_24 = inject_24(w, size_k, size_n)

    # Quantize
    w_24_ref, q_w_24, s, g_idx, rand_perm = quantize_weights(w_24,
                                                             num_bits,
                                                             group_size,
                                                             act_order=False)

    # Packs to marlin 2:4
    marlin_24_q_w_comp, marlin_24_s, meta = pack_to_marlin_24(
        q_w_24, s, num_bits, group_size
    )

    # Create result
    res_list = [w_24_ref, marlin_24_q_w_comp, meta, marlin_24_s]
    for i in range(len(res_list)):
        res_list[i] = res_list[i].to(w.device)

    return res_list


class MarlinWorkspace:

    def __init__(self, out_features, min_thread_n, max_parallel):
        assert (out_features % min_thread_n == 0), (
            "out_features = {} is undivisible by min_thread_n = {}".format(
                out_features, min_thread_n))

        max_workspace_size = ((out_features // min_thread_n) * max_parallel)

        self.scratch = torch.zeros(max_workspace_size,
                                   dtype=torch.int,
                                   device="cuda")
