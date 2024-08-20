import torch
import numpy as np
from typing import Tuple, Dict, List

import torchao.sparsity.marlin.utils as utils
from torchao.sparsity.marlin.utils import const


__all__ = [
    "inject_24",
    "marlin_24_workspace",
    "pack_to_marlin_24",
    "unpack_from_marlin_24",
]


def inject_24(w, size_k, size_n):
    assert w.shape == (size_k, size_n)
    mask = utils.mask_creator(w.t()).t().cuda().bool()
    return (mask * w).contiguous(), mask.contiguous()


def marlin_24_workspace(
        out_features: int, 
        min_thread_n: int = const.MIN_THREAD_N, 
        max_parallel: int = const.MAX_PARALLEL
    ) -> torch.Tensor:
    """Creates a workspace for marlin 2:4 quantization. The workspace is used to coordinate the locks 
    during the execution of the kernel.
    
    Args:
        out_features (int): The number of output features.
        min_thread_n (int, optional): The minimum number of threads per block. Defaults to `MARLIN_24_MIN_THREAD_N`.
        max_parallel (int, optional): The maximum number of parallel threads. Defaults to `MARLIN_24_MAX_PARALLEL`.

    Returns:
        torch.Tensor: The workspace tensor fully initialized with zeros.
    """
    assert (out_features % min_thread_n == 0), f"out_features = {out_features}, min_thread_n = {min_thread_n}"
    max_workspace_size = ((out_features // min_thread_n) * max_parallel)
    return torch.zeros(max_workspace_size, dtype=torch.int, device="cuda")


def pack_to_marlin_24(
        q_w_24: torch.Tensor, 
        scales: torch.Tensor, 
        num_bits: int, 
        group_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    in_features, out_features = q_w_24.shape

    # Compress quantized weight
    q_w_24_comp, meta = _compress_quantized_24_weight(
        q_w_24, in_features, out_features, num_bits
    )

    in_features_comp = in_features // 2

    # Reformat to marlin
    marlin_24_q_w_comp = _to_marlin_weights(
        q_w_24_comp, in_features_comp, out_features, num_bits
    )

    reverse = _from_marlin_weights(
        marlin_24_q_w_comp, in_features_comp, out_features, num_bits
    )
    assert torch.equal(reverse, q_w_24_comp)

    marlin_24_s = _to_marlin_scales(
        scales, in_features, out_features, group_size, num_bits
    )

    return marlin_24_q_w_comp, marlin_24_s, meta


def unpack_from_marlin_24(
        q_w_24_comp: torch.Tensor, 
        scales: torch.Tensor, 
        meta: torch.Tensor, 
        original_shape: torch.Size,
        group_size: int,
        num_bits: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    in_features, out_features = original_shape

    # Unpacks the scales 
    unpacked_scales = _from_marlin_scale(
        scales, *original_shape, group_size, num_bits
    )

    in_features_comp = in_features // 2

    # Unpacks the weights
    unpacked_q_w_24_comp = _from_marlin_weights(
        q_w_24_comp, in_features_comp, out_features, num_bits
    )

    # Decompress quantized weight
    unpacked_q_w_24 = _decompress_quantized_24_weight(
        unpacked_q_w_24_comp, meta, in_features_comp, out_features, num_bits
    )

    return unpacked_q_w_24, unpacked_scales


def _compress_quantized_24_weight(q_24, size_k, size_n, num_bits):
    assert q_24.shape == (size_k, size_n)

    # Remove zp to normalize over 0
    max_q_val = (1 << num_bits) - 1
    zp = (max_q_val + 1) // 2
    q_24_no_zp = q_24 - zp

    # Compress
    q_24_no_zp = q_24_no_zp.t().contiguous()
    q_24_no_zp_comp, meta = utils.sparse_semi_structured_from_dense_cutlass(q_24_no_zp)
    q_24_no_zp_comp = q_24_no_zp_comp.t().contiguous()

    # Restore zp
    q_24_comp = q_24_no_zp_comp + zp

    # Resize meta to its actual shape (without moving any data)
    meta = meta.resize_(meta.shape[1] // 2, meta.shape[0] * 2)

    return q_24_comp, meta


# TODO(diogo): WIP
def _decompress_quantized_24_weight(q_24_comp, meta, size_k, size_n, num_bits) -> torch.Tensor:
    assert q_24_comp.shape == (size_k, size_n)

    # Resize meta back to its original shape
    meta = meta.resize_(meta.shape[1] // 2, meta.shape[0] * 2)

    # Remove zp to normalize over 0
    max_q_val = (1 << num_bits) - 1
    zp = (max_q_val + 1) // 2
    q_24_no_zp_comp = q_24_comp - zp

    # Decompress
    q_24_no_zp_comp = q_24_no_zp_comp.t().contiguous()
    q_24_no_zp = utils.sparse_semi_structured_to_dense_cutlass(q_24_no_zp_comp, meta)
    q_24_no_zp = q_24_no_zp.t().contiguous()

    # Restore zp
    q_24 = q_24_no_zp + zp

    return q_24


def _to_marlin_weights(q_w, size_k, size_n, num_bits):
    # Permute
    q_w = utils.marlin_permute_weights(q_w, size_k, size_n, marlin_24_perm[num_bits])

    # Pack
    pack_factor = utils.get_pack_factor(num_bits)
    orig_device = q_w.device

    q_w = q_w.cpu().numpy().astype(np.uint32)

    q_packed = np.zeros((q_w.shape[0], q_w.shape[1] // pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_packed |= q_w[:, i::pack_factor] << num_bits * i

    q_packed = torch.from_numpy(q_packed.astype(np.int32)).to(orig_device)

    return q_packed


def _from_marlin_weights(q_packed, size_k, size_n, num_bits, tile=const.TILE):
    reverse_perm = reverse_marlin_24_perm[num_bits]

    pack_factor = utils.get_pack_factor(num_bits)
    orig_device = q_packed.device

    # Unpack
    q_packed = q_packed.cpu().numpy().astype(np.uint32)
    q_w_unpacked = np.zeros((q_packed.shape[0], q_packed.shape[1] * pack_factor), dtype=np.uint32)
    for i in range(pack_factor):
        q_w_unpacked[:, i::pack_factor] = (q_packed >> (num_bits * i)) & ((1 << num_bits) - 1)

    q_w_unpacked = torch.from_numpy(q_w_unpacked.astype(np.int32)).to(orig_device)

    q_w_comp = utils.reverse_marlin_permute_weights(
        q_w_unpacked, size_k, size_n, reverse_perm
    )

    return q_w_comp


def _to_marlin_scales(s, size_k, size_n, group_size, num_bits: int):
    if group_size < size_k and group_size != -1:
        perms = marlin_24_scale_perm[num_bits]
        s = s.reshape((-1, len(perms)))[:, perms]
    else:
        perms = marlin_24_scale_perm_single[num_bits]
        s = s.reshape((-1, len(perms)))[:, perms]
    s = s.reshape((-1, size_n)).contiguous()
    return s


def _from_marlin_scale(s, size_k, size_n, group_size, num_bits: int) -> torch.Tensor:
    s = s.reshape((-1, size_n)).contiguous()

    if group_size < size_k and group_size != -1:
        reverse_perms = reverse_marlin_24_scale_perm[num_bits]
        s = s.reshape((-1, len(reverse_perms)))[:, reverse_perms]
    else:
        reverse_perms = reverse_marlin_24_scale_perm_single[num_bits]
        s = s.reshape((-1, len(reverse_perms)))[:, reverse_perms]

    return s.reshape(-1).contiguous()


# Contains the permutations for marlin 2:4 quantization
marlin_24_perm: Dict[int, torch.Tensor] = {}
marlin_24_scale_perm: Dict[int, List[int]] = {}
marlin_24_scale_perm_single: Dict[int, List[int]] = {}

# Contains the reverse permutations for marlin 2:4 quantization
reverse_marlin_24_perm: Dict[int, torch.Tensor] = {}
reverse_marlin_24_scale_perm: Dict[int, List[int]] = {}
reverse_marlin_24_scale_perm_single: Dict[int, List[int]] = {}

for num_bits in const.SUPPORTED_NUM_BITS:
    perm_24, scale_perm_24, scale_perm_single_24 = utils.get_perms_24(num_bits)

    marlin_24_perm[num_bits] = perm_24
    marlin_24_scale_perm[num_bits] = scale_perm_24
    marlin_24_scale_perm_single[num_bits] = scale_perm_single_24

    reverse_marlin_24_perm[num_bits] = perm_24.argsort()
    reverse_marlin_24_scale_perm[num_bits] = torch.tensor(scale_perm_24).argsort()
    reverse_marlin_24_scale_perm_single[num_bits] = torch.tensor(scale_perm_single_24).argsort()
