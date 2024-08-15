import torch
from typing import Tuple


# The code in this file has been adapted from the following source:
# https://github.com/IST-DASLab/Sparse-Marlin/blob/main/marlin/_semi_structured_conversions.py


def pack_to_sparse_marlin_24(
        weight: torch.Tensor, 
        scales: torch.Tensor, 
        n_tiles: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Pack a fake-quantized linear layer into this actual Marlin representation.
    @linear: fake-quantized `torch.nn.Linear` layer to convert (must be of type `torch.half`)
    @scales: corresponding quantization scales of shape `(infeatures, groups)`
    """
    import numpy as np
    in_features, out_features = weight.shape
    group_size = 128
    tile = n_tiles
    s = scales
    w = weight

    w = w.reshape((group_size, -1, out_features))
    w = w.permute(1, 0, 2)
    w = w.reshape((in_features, out_features)).contiguous()
    s = s.reshape((-1, len(scale_perm)))[:, scale_perm]

    mask = _mask_creator(w.T).cuda().bool()
    w = mask * w.T
    w, meta = _sparse_semi_structured_from_dense_cutlass(w)
    w = w.t()

    in_features = in_features // 2

    s = s.reshape((-1, out_features)).contiguous()
    w = w.reshape((in_features // tile, tile, out_features // tile, tile))
    w = w.permute((0, 2, 1, 3))
    w = w.reshape((in_features // tile, out_features * tile))
    res = w
    res = res.reshape((-1, perm.numel()))[:, perm].reshape(res.shape)

    # TODO(diogo): check if I can do this with torch?
    q = np.zeros((res.shape[0], res.shape[1] // 8), dtype=np.uint32)
    res = res.cpu().numpy().astype(np.uint32)
    for i in range(8):
        q |= res[:, i::8] << 4 * i

    q = torch.from_numpy(q.astype(np.int32)).to(w.device)
    q[:, :] = q.to(weight.device)
    s[:, :] = s.to(scales.device)
    meta[:, :] = meta.to(meta.device)

    return q, s, meta


def unpack_from_sparse_marlin_24(q: torch.Tensor, s: torch.Tensor, meta: torch.Tensor, n_tiles: int, initial_shape: torch.Size) -> Tuple[torch.Tensor, torch.Tensor]:

    def unpack_scales(scales: torch.Tensor):
        
        # Step 1: Reverse the final reshape
        # Original reshaped scales were of shape (-1, out_features)
        s_unpacked = scales.reshape((-1, len(scale_perm)))

        # Step 2: Reverse the permutation
        reverse_perm = torch.tensor(scale_perm).argsort()
        s_unpacked = s_unpacked[:, reverse_perm]

        # Step 3: Reverse the initial reshape
        # Flatten the scales back to 1D
        s_unpacked = s_unpacked.reshape(-1)

        return s_unpacked

    def unpack_weights(q: torch.Tensor, tile: int, meta: torch.Tensor, initial_shape: Tuple[int, int]):
        import numpy as np

        res_ = np.zeros((q.shape[0], q.shape[1] * 8), dtype=np.uint32)
        for i in range(8):
            res_[:, i::8] = (q.cpu().numpy() >> (4 * i)) & 0xF
        res_ = torch.from_numpy(res_.astype(np.int32)).to(q.device)

        # Step 2: Reverse the permutation
        res_ = res_.reshape((-1, perm.numel()))[:, torch.argsort(perm)].reshape(res_.shape)

        # Step 3: Reverse the reshape and permutation applied before
        in_features, out_features = initial_shape
        in_features = in_features // 2

        w = res_.reshape((in_features // tile, out_features // tile, tile, tile))
        w = w.permute((0, 2, 1, 3))
        w = w.reshape((in_features, out_features))
        w = w.t()

        w_unpacked = _sparse_semi_structured_to_dense_cutlass(w, meta)
        w_unpacked = w_unpacked.t()

        return w_unpacked

    return unpack_weights(q, n_tiles, meta, initial_shape), unpack_scales(s)


# This function converts dense matrix into sparse semi-structured
# representation, producing "compressed" matrix, in the layout used by
# CUTLASS backend, and corresponding metadata matrix.
def _sparse_semi_structured_from_dense_cutlass(dense):
    if dense.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional dense tensor, got {dense.dim()}-dimensional tensor"
        )

    m, k = dense.shape
    device = dense.device

    meta_dtype = torch.int8
    if dense.dtype == torch.int8:
        meta_dtype = torch.int32
    elif dense.dtype in [torch.half, torch.bfloat16, torch.float, torch.int32]:
        meta_dtype = torch.int16
    else:
        raise RuntimeError(f"Invalid datatype {dense.dtype} of dense matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4
    if quadbits_per_meta_elem not in (4, 8):
        raise RuntimeError("Invalid number of elements per meta element calculated")

    if meta_dtype == torch.int32:
        if m % 16 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 16"
            )
    else:
        if m % 32 != 0:
            raise RuntimeError(
                f"Number of rows of dense matrix {m} must be divisible by 32"
            )
    if k % (4 * quadbits_per_meta_elem) != 0:
        raise RuntimeError(
            f"Number of columns of dense matrix {k} must be divisible by {4 * quadbits_per_meta_elem}"
        )

    if dense.dtype != torch.float:
        ksparse = 4
        dense_4 = dense.view(-1, k // ksparse, ksparse)
        m0, m1, m2, m3 = (dense_4 != 0).unbind(-1)
    else:
        ksparse = 2
        dense_2 = dense.view(-1, k // ksparse, ksparse)
        m0, m2 = m1, m3 = (dense_2 != 0).unbind(-1)
    meta_ncols = k // (ksparse * quadbits_per_meta_elem)

    # Encoding quadruples of True/False values as follows:
    #     [True,  True,  False, False] -> 0b0100
    #     [True,  False, True,  False] -> 0b1000
    #     [False, True,  True,  False] -> 0b1001
    #     [True,  False, False, True ] -> 0b1100
    #     [False, True,  False, True ] -> 0b1101
    #     [False, False, True,  True ] -> 0b1110
    # Thus, lower two bits in the encoding are index of the True value
    # at the lowest index in the quadruple, and the higher two bits in
    # the encoding are index of the other True value in the quadruple.
    # In case there are less than two True values, than False value or
    # values at some index or indices are considered True for the
    # encoding.  In case there are more than two True values, then the
    # excess True value(s) at some indices are considered False for
    # the encoding.  The exact encodings used for these cases are as
    # follows:
    #     [False, False, False, False] -> 0b1110
    #     [False, False, False, True ] -> 0b1110
    #     [False, False, True,  False] -> 0b1110
    #     [False, True,  False, False] -> 0b1001
    #     [False, True,  True,  True ] -> 0b1101
    #     [True,  False, False, False] -> 0b1000
    #     [True,  False, True,  True ] -> 0b1100
    #     [True,  True,  False, True ] -> 0b0100
    #     [True,  True,  True,  False] -> 0b0100
    #     [True,  True,  True,  True ] -> 0b0100
    # These particular encodings are chosen, with the help of Espresso
    # logic minimizer software, for the purpose of minimization of
    # corresponding Boolean functions, that translate non-zero flags
    # into encoding bits.  Note also possible choices for the first
    # and last of these encodings were limited only to (0b0100,
    # 0b1110), in order to produce valid encodings for 1:2 sparsity
    # case.

    expr0 = m0 & m1
    expr1 = ~m0 & m1
    expr2 = ~m0 & ~m1
    bit0 = expr1
    bit1 = expr2
    bit2 = expr0 | expr2 | m3
    bit3 = expr1 | ~m1
    idxs0 = bit0 | (bit1.to(torch.int64) << 1)
    idxs1 = bit2 | (bit3.to(torch.int64) << 1)

    if dense.dtype != torch.float:
        sparse0 = dense_4.gather(-1, idxs0.unsqueeze(-1))  # type: ignore[possibly-undefined]
        sparse1 = dense_4.gather(-1, idxs1.unsqueeze(-1))
        sparse = torch.stack((sparse0, sparse1), dim=-1).view(m, k // 2)
    else:
        sparse = dense_2.gather(-1, idxs0.unsqueeze(-1) // 2).view(m, k // 2)  # type: ignore[possibly-undefined]

    meta_4 = idxs0 | (idxs1 << 2)
    meta_n = meta_4.view((-1, meta_ncols, quadbits_per_meta_elem)).to(meta_dtype)

    if quadbits_per_meta_elem == 4:
        meta = (
            meta_n[:, :, 0]
            | (meta_n[:, :, 1] << 4)
            | (meta_n[:, :, 2] << 8)
            | (meta_n[:, :, 3] << 12)
        )
    elif quadbits_per_meta_elem == 8:
        meta = (
            meta_n[:, :, 0]
            | (meta_n[:, :, 1] << 4)
            | (meta_n[:, :, 2] << 8)
            | (meta_n[:, :, 3] << 12)
            | (meta_n[:, :, 4] << 16)
            | (meta_n[:, :, 5] << 20)
            | (meta_n[:, :, 6] << 24)
            | (meta_n[:, :, 7] << 28)
        )

    # Reorder meta tensor elements.
    meta_reordered = meta.new_empty((m * meta_ncols,))  # type: ignore[possibly-undefined]
    meta_offsets = _calculate_meta_reordering_scatter_offsets(
        m, meta_ncols, meta_dtype, device
    )
    meta_reordered.scatter_(0, meta_offsets, meta.view(-1))

    return (sparse, meta_reordered.view(m, meta_ncols))


# This function performs reverse of the function above - it
# reconstructs dense matrix from a pair of "compressed" matrix, given
# in the layout used by CUTLASS backend, and accompanying metadata
# matrix.
def _sparse_semi_structured_to_dense_cutlass(sparse, meta_reordered):
    if sparse.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional sparse tensor, got {sparse.dim()}-dimensional tensor"
        )

    m, k = sparse.shape
    device = sparse.device

    if meta_reordered.dim() != 2:
        raise RuntimeError(
            f"Expected 2-dimensional meta tensor, got {meta_reordered.dim()}-dimensional tensor"
        )
    if meta_reordered.device != device:
        raise RuntimeError(
            f"Expected meta matrix to be on {device} device, got matrix on {meta_reordered.device} device"
        )

    meta_dtype = meta_reordered.dtype
    if meta_dtype not in (torch.int16, torch.int32):
        raise RuntimeError(f"Invalid datatype {meta_dtype} of meta matrix")
    quadbits_per_meta_elem = meta_dtype.itemsize * 8 // 4

    if sparse.dtype != torch.float:
        ksparse = 4
    else:
        ksparse = 2

    meta_nrows, meta_ncols = meta_reordered.shape
    if meta_nrows != m:
        raise RuntimeError(
            f"Number of rows of meta matrix {meta_nrows} must be equal to number of columns of spase matrix {m}"
        )
    if meta_ncols * ksparse * quadbits_per_meta_elem != 2 * k:
        raise RuntimeError(
            f"Number of columns of sparse matrix {k} different from the {meta_ncols * ksparse * quadbits_per_meta_elem // 2}, "
            "expected according to the number of columns of meta matrix"
        )

    # Undo meta tensor elements reordering.
    meta_offsets = _calculate_meta_reordering_scatter_offsets(
        m, meta_ncols, meta_dtype, device
    )
    meta = torch.gather(meta_reordered.view(-1), 0, meta_offsets).view(m, meta_ncols)

    # Unpack sparse tensor back to original dense tensor, using
    # information provided by meta tensor.  Note that torch.float
    # datatype is handled pretty much the same as
    # torch.half/torch.bfloat16, as metadata for a pair of torch.float
    # value is encoded as if underlying 8 bytes contain four
    # torch.half/torch.bfloat16 values, where either first two or last
    # two are zeros.
    meta_2 = torch.empty(
        (m, meta_ncols, 2 * quadbits_per_meta_elem),
        dtype=meta_dtype,
        device=device,
    )
    if quadbits_per_meta_elem == 4:
        meta_2[:, :, 0] = meta & 0b11
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
    elif quadbits_per_meta_elem == 8:
        meta_2[:, :, 0] = meta & 0b11
        meta_2[:, :, 1] = (meta >> 2) & 0b11
        meta_2[:, :, 2] = (meta >> 4) & 0b11
        meta_2[:, :, 3] = (meta >> 6) & 0b11
        meta_2[:, :, 4] = (meta >> 8) & 0b11
        meta_2[:, :, 5] = (meta >> 10) & 0b11
        meta_2[:, :, 6] = (meta >> 12) & 0b11
        meta_2[:, :, 7] = (meta >> 14) & 0b11
        meta_2[:, :, 8] = (meta >> 16) & 0b11
        meta_2[:, :, 9] = (meta >> 18) & 0b11
        meta_2[:, :, 10] = (meta >> 20) & 0b11
        meta_2[:, :, 11] = (meta >> 22) & 0b11
        meta_2[:, :, 12] = (meta >> 24) & 0b11
        meta_2[:, :, 13] = (meta >> 26) & 0b11
        meta_2[:, :, 14] = (meta >> 28) & 0b11
        meta_2[:, :, 15] = (meta >> 30) & 0b11

    dense_offsets = meta_2.view(-1) + (
        torch.arange(0, 2 * m * k // ksparse, device=device) * 4
    ).view(-1, 1).repeat(1, 2).view(-1)

    dense = torch.zeros((m * 2 * k,), dtype=sparse.dtype, device=device)
    if sparse.dtype != torch.float:
        # dense.scatter_(0, dense_offsets, sparse.view(-1))
        dense.scatter_(0, dense_offsets, sparse.reshape(-1))
    else:
        dense.view(torch.half).scatter_(
            0, dense_offsets, sparse.view(torch.half).view(-1)
        )

    return dense.view(m, 2 * k)


# This is PyTorch implementation of main part of reorder_meta()
# function, from tools/util/include/cutlass/util/host_reorder.h file
# of CUTLASS source tree.  Furthermore, CUTLASS template for sparse
# GEMM decides upon layout of this matrix, and at the moment for the
# sparse GEMM executed on tensor cores, this is layout described by
# ColumnMajorInterleaved<2> data structure, in
# include/cutlass/layout/matrix.h of CUTLASS source tree.  The
# reordering of meta matrix into meta_reordered matrix calculated
# according to these segments of CUTLASS code is re-implemented here.
# Note that this calculation produces offsets for scattering metadata
# matrix elements into reordered metadata matrix elements (or,
# equivalently, for gathering reordered metadata matrix element back
# into metadata matrix elements).
def _calculate_meta_reordering_scatter_offsets(m, meta_ncols, meta_dtype, device):
    dst_rows = torch.arange(0, m, device=device)[:, None].repeat(1, meta_ncols)
    dst_cols = torch.arange(0, meta_ncols, device=device).repeat(m, 1)

    # Reorder the rows, then swizzle the 2x2 blocks.
    group_x = 64
    group_y = 32 if meta_dtype.itemsize == 2 else 16

    dst_rows = (
        dst_rows // group_x * group_x
        + (dst_rows % 2) * 2
        + (dst_rows % 8) // 4
        + ((dst_rows % group_y) % 4) // 2 * 32
        + ((dst_rows % group_x) // 8) * 4
    )

    topright = ((dst_rows % 2 == 0) & (dst_cols % 2 == 1)).to(torch.int8)
    bottomleft = ((dst_rows % 2 == 1) & (dst_cols % 2 == 0)).to(torch.int8)
    dst_rows += topright - bottomleft
    dst_cols -= topright - bottomleft

    # Assumed that meta tensor is to be stored in CUTLASS
    # InterleavedColumnMajor layout, and reverse engineered
    # corresponding code to store values into this tensor.
    interleave = 2
    cols_maj = dst_cols // interleave
    cols_min = dst_cols % interleave
    return (cols_maj * m * interleave + dst_rows * interleave + cols_min).view(-1)


def _mask_creator(tensor):
    """
    Class for creating N:M sparsity masks.
    Masks will be created using the N:M ratio, where for every block of M weights,
    N will be pruned based on ranked weight value. Each mask will correspond to the given tensor.

    :param N: The number of weights in a group to keep
    :param M: The size of a weight group
    """
    N = 2
    M = 4

    mask = None
    # for i, tensor in enumerate(tensors):
    if tensor.numel() % M != 0:
        raise ValueError(
            f"Tensor of size {tensor.shape} can't be evenly divided into " f"{M} groups"
        )

    num_groups = tensor.numel() // M

    # N:M sparsity for linear layers
    tensor_temp = tensor.detach().abs().reshape(num_groups, M)
    index = torch.argsort(tensor_temp, dim=1)[:, : int(M - N)]

    w_b = torch.ones(tensor_temp.shape, device=tensor_temp.device)
    mask = w_b.scatter_(dim=1, index=index, value=0).reshape(tensor.shape)

    return mask


def _get_perms_2_4():
    import numpy as np
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        col_o = col // 2
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1,
            ]:
                perm1.append(16 * row + col_o * 256 + 8 * (col % 2) + 4 * block)
        for j in range(4):
            perm.extend([p + 1 * j for p in perm1])
    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(8):
        scale_perm.extend([i * 8 + j for j in [0, 4, 1, 5, 2, 6, 3, 7]])
    scale_perm_single = []
    for i in range(8):
        scale_perm_single.extend([8 * i + j for j in [0, 1, 2, 3, 4, 5, 6, 7]])
    return perm, scale_perm, scale_perm_single

# Precompute permutations for Marlin weight and scale shuffling
perm, scale_perm, scale_perm_single = _get_perms_2_4()
