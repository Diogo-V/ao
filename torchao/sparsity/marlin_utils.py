import numpy
import torch
from typing import Tuple

from torchao.sparsity.ccc import (
    quantize_weights
)
from torchao.sparsity.marlin import pack_to_marlin_24, inject_24


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
