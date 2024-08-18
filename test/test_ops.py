import itertools

import torchao

import torch
from torch.testing._internal.common_utils import (
    TestCase,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
)
from torch.testing._internal.optests import opcheck
from torchao.utils import is_fbcode, TORCH_VERSION_AT_LEAST_2_5
from torchao.prototype.quant_llm import from_scaled_tc_fpx
import pytest

if is_fbcode():
    pytest.skip("Skipping the test in fbcode since we don't have TARGET file for kernels")

try:
    import torchao.ops
except RuntimeError:
    pytest.skip("torchao.ops not available")

from torchao.sparsity.utils import mask_creator
from torchao.sparsity.marlin import (
    pack_to_sparse_marlin_24,
    marlin_24_mm,
    fp16_to_int4_marlin_format
)
from torchao.quantization.utils import (
    get_groupwise_affine_qparams,
    groupwise_affine_dequantize_tensor_from_qparams,
    groupwise_affine_quantize_tensor_from_qparams,
    pack_tinygemm_scales_and_zeros,
    unpack_tinygemm_scales_and_zeros,
)


class TestOps(TestCase):
    def _create_fpx_inputs(self, ebits: int, mbits: int, BS: int, OC: int, IC: int, device):
        # Randomly initialize each byte
        nbits = 1 + ebits + mbits
        fpx_weight = torch.randint(256, (OC, IC // 8 * nbits), dtype=torch.uint8)
        scale = torch.rand(OC).half() + 0.5
        fp16_act = torch.rand(BS, IC).half() + 0.5
        return fpx_weight.to(device), scale.to(device), fp16_act.to(device)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @parametrize("ebits,mbits", [(3, 2), (2, 2)])
    def test_quant_llm_linear(self, ebits, mbits):
        BS = 2
        OC = 256
        IC = 256
        splitK = 1
        fpx_weight, scale, fp16_act = self._create_fpx_inputs(ebits, mbits, BS, OC, IC, "cuda")

        # smoke test
        torchao.ops.quant_llm_linear(ebits, mbits, fp16_act, fpx_weight, scale, splitK)

        # comprehensive testing
        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(torch.ops.torchao.quant_llm_linear, (ebits, mbits, fp16_act, fpx_weight, scale, splitK), test_utils=test_utils)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @parametrize("BS,OC,IC,splitK", [(1, 2048, 4096, 5), (2, 8192, 8192, 6)])
    @parametrize("ebits,mbits", [(3, 2), (2, 2)])
    def test_quant_llm_linear_correctness(self, ebits, mbits, BS, OC, IC, splitK):
        # adapted from https://github.com/usyd-fsalab/fp6_llm/blob/5df6737cca32f604e957e3f63f03ccc2e4d1df0d/tests/python/kernel_test_fpx.py
        fpx_weight, scale, fp16_act = self._create_fpx_inputs(ebits, mbits, BS, OC, IC, "cuda")

        results_fpx = torchao.ops.quant_llm_linear(ebits, mbits, fp16_act, fpx_weight, scale, splitK)

        fp16_weight = from_scaled_tc_fpx(fpx_weight, ebits, mbits, scale).half()
        results_fp16 = fp16_act @ fp16_weight.T

        error = (results_fpx - results_fp16).abs().mean()
        gt = results_fp16.abs().mean()
        relative_error = error / gt
        assert relative_error < 1e-3

instantiate_parametrized_tests(TestOps)


## Tests for `tensor_core_layout`
kTileSizeN = 8
kTileSizeK = 16

SHAPES = [
    (4096, 4096),
    # Llama 2 GEMM shapes
    (4096, 11008),
    (11008, 4096),
    # Llama 3 GEMM shapes
    (4096, 14336),
    (14336, 4096),
]
INNERKTILES = [2, 4, 8]
QGROUP_SIZES = [32, 64, 128, 256]
TEST_CONFIGS_UNPACK = list(itertools.product(SHAPES, INNERKTILES))
TEST_CONFIGS_DEQUANT = list(itertools.product(SHAPES, INNERKTILES, QGROUP_SIZES))

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# @pytest.mark.skipif(TORCH_VERSION_AT_LEAST_2_5, reason="weight packing is updated in 2.5+")
@pytest.mark.parametrize("shape, inner_k_tiles", TEST_CONFIGS_UNPACK, ids=str)
def test_unpack_tensor_core_tiled_layout_correctness(shape, inner_k_tiles):
    N, K = shape
    assert K % (inner_k_tiles * kTileSizeK) == 0 and N % kTileSizeN == 0

    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    if TORCH_VERSION_AT_LEAST_2_5:
        t = (t[::, ::2] << 4 | t[::, 1::2]).to(torch.uint8)
    packed_w = torch.ops.aten._convert_weight_to_int4pack(t, inner_k_tiles)
    unpacked = torchao.ops.unpack_tensor_core_tiled_layout(packed_w, inner_k_tiles)
    if TORCH_VERSION_AT_LEAST_2_5:
        unpacked = (unpacked[::, ::2] << 4 | unpacked[::, 1::2]).to(torch.uint8)
    assert torch.equal(t, unpacked)

# TODO: Fix "test_aot_dispatch_dynamic" test failure
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# @pytest.mark.skipif(TORCH_VERSION_AT_LEAST_2_5, reason="weight packing is updated in 2.5+")
@pytest.mark.parametrize("shape, inner_k_tiles", TEST_CONFIGS_UNPACK , ids=str)
def test_unpack_tensor_core_tiled_layout_op(shape, inner_k_tiles):
    test_utils = [
        "test_schema",
        "test_autograd_registration",
        "test_faketensor",
    ]

    # TODO: Figure out why test fails unless torch >= 2.5
    if TORCH_VERSION_AT_LEAST_2_5:
        test_utils.append("test_aot_dispatch_dynamic")

    t = torch.randint(0, 16, dtype=torch.int, size=shape, device="cuda")
    if TORCH_VERSION_AT_LEAST_2_5:
        t = (t[::, ::2] << 4 | t[::, 1::2]).to(torch.uint8)
    packed_w = torch.ops.aten._convert_weight_to_int4pack(t, inner_k_tiles)

    opcheck(
        torch.ops.torchao.unpack_tensor_core_tiled_layout,
        (packed_w, inner_k_tiles),
        test_utils=test_utils,
    )

def dequant_ref(q, scales, zeros, group_size, nbits=4, dtype=torch.bfloat16):
    n, k = q.shape
    assert q.dtype == torch.int

    n_groups = k // group_size
    assert scales.shape[0] == n and scales.shape[1] == n_groups
    assert scales.shape == zeros.shape

    midpoint = 2 ** (nbits - 1)

    #Convert fron u4 -> s4 and upcast to bfloat16
    q = q.sub(midpoint).to(dtype)

    # Dequantize
    q = q.reshape(-1, group_size)
    dq = q * scales.reshape(-1, 1) + zeros.reshape(-1, 1)

    return dq.reshape(n, k)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# @pytest.mark.skipif(TORCH_VERSION_AT_LEAST_2_5, reason="weight packing is updated in 2.5+")
@pytest.mark.parametrize("shape, inner_k_tiles, group_size", TEST_CONFIGS_DEQUANT, ids=str)
def test_dequantize_tensor_core_tiled_layout_correctness_quant_dequant(shape, inner_k_tiles, group_size):
    n, k = shape
    dtype = torch.bfloat16

    device = "cuda"

    t = torch.randn(n, k, dtype=dtype, device=device)
    scales, zeros = get_groupwise_affine_qparams(t, n_bit=4, groupsize=group_size, dtype=dtype)

    # Quantize
    q = groupwise_affine_quantize_tensor_from_qparams(
        t, scales, zeros, n_bit=4, groupsize=group_size
    )

    # Pack to tensor core layout
    packed = torch.ops.aten._convert_weight_to_int4pack(q, inner_k_tiles)
    scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros)
    q_groups = k // group_size
    assert scales_and_zeros.shape == torch.Size([q_groups, n, 2])

    # Dequantize 'ao' ref
    dq_ao = groupwise_affine_dequantize_tensor_from_qparams(
        q, scales, zeros, n_bit=4, groupsize=group_size
    )

    # Dequantize by passing in an identity matrix as the activation
    a_eye = torch.eye(k, device=device, dtype=dtype)
    dq_id = torch.ops.aten._weight_int4pack_mm(
        a_eye,
        packed,
        group_size,
        scales_and_zeros,
    ).t()

    # Actual operation to test
    dq_op = torchao.ops.dequantize_tensor_core_tiled_layout(packed, scales_and_zeros, group_size, inner_k_tiles)

    # Compare results
    diff_ao_id = (dq_id - dq_ao).abs().max()
    diff_op_id = (dq_op - dq_id).abs().max()
    diff_op_ao = (dq_op - dq_ao).abs().max()

    # There are slight numerical differences when dequantizing with an identity matrix when compared to `groupwise_affine_dequantize`
    # Since the `dequantize_tensor_core_layout` kernel relies on the same underlying bit twiddling tricks for fast
    # conversion from u4 -> s4 -> bf16, the identity matrix dequant hack and `dequantize_tensor_core_layout` are
    # expected to give same results, while both will have similar numerical differences to `groupwise_affine_dequantize`.

    # Test that the `dequant` kernel gives same results as identity matrix-based dequant
    assert diff_op_id == 0

    # Test that the `dequant` kernel gives same numerical diffs as the `groupwise_affine_dequantize` when compared against the identity matrix
    assert diff_op_ao == diff_ao_id

    assert diff_op_ao < 1e-1

# This test differs from one above in that it uses `unpack_tensor_core_tiled_layout` to unpack then dequantize
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# @pytest.mark.skipif(TORCH_VERSION_AT_LEAST_2_5, reason="weight packing is updated in 2.5+")
@pytest.mark.parametrize("shape, inner_k_tiles, group_size", TEST_CONFIGS_DEQUANT, ids=str)
def test_dequantize_tensor_core_tiled_layout_correctness_unpack_and_dequant(shape, inner_k_tiles, group_size):
    n, k = shape
    dtype = torch.bfloat16
    device = "cuda"

    # Quantize and pack
    t = torch.randn(n, k, dtype=dtype, device=device)
    scales, zeros = get_groupwise_affine_qparams(t, n_bit=4, groupsize=group_size, dtype=dtype)
    q = groupwise_affine_quantize_tensor_from_qparams(
        t, scales, zeros, n_bit=4, groupsize=group_size
    )

    packed = torch.ops.aten._convert_weight_to_int4pack(q, inner_k_tiles)
    scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros)

    # Unpack and dequantize
    unpacked = torchao.ops.unpack_tensor_core_tiled_layout(packed, inner_k_tiles)
    if TORCH_VERSION_AT_LEAST_2_5:
        unpacked = (unpacked[::, ::2] << 4 | unpacked[::, 1::2]).to(torch.uint8)

    dq_ao = groupwise_affine_dequantize_tensor_from_qparams(
        unpacked, scales, zeros, n_bit=4, groupsize=group_size
    )

    # Dequantize by passing in an identity matrix as the activation
    a_eye = torch.eye(k, device=device, dtype=dtype)
    dq_id = torch.ops.aten._weight_int4pack_mm(
        a_eye,
        packed,
        group_size,
        scales_and_zeros,
    ).t()

    # Actual operation to test
    dq_op = torchao.ops.dequantize_tensor_core_tiled_layout(packed, scales_and_zeros, group_size, inner_k_tiles)

    # Compare results
    diff_ao_id = (dq_id - dq_ao).abs().max()
    diff_op_id = (dq_op - dq_id).abs().max()
    diff_op_ao = (dq_op - dq_ao).abs().max()

    # There are slight numerical differences when dequantizing with an identity matrix when compared to `groupwise_affine_dequantize`
    # Since the `dequantize_tensor_core_layout` kernel relies on the same underlying bit twiddling tricks for fast
    # conversion from u4 -> s4 -> bf16, the identity matrix dequant hack and `dequantize_tensor_core_layout` are
    # expected to give same results, while both will have similar numerical differences to `groupwise_affine_dequantize`.

    # Test that the `dequant` kernel gives same results as identity matrix-based dequant
    assert diff_op_id == 0

    # Test that the `dequant` kernel gives same numerical diffs as the `groupwise_affine_dequantize` when compared against the identity matrix
    assert diff_op_ao == diff_ao_id

    assert diff_op_ao < 1e-1

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
# @pytest.mark.skipif(TORCH_VERSION_AT_LEAST_2_5, reason="weight packing is updated in 2.5+")
@pytest.mark.parametrize("shape, inner_k_tiles, group_size", TEST_CONFIGS_DEQUANT, ids=str)
def test_dequantize_tensor_core_tiled_layout_op(shape, inner_k_tiles, group_size):
    n, k = shape
    device = "cuda"

    q = torch.randint(0, 16, shape, dtype=torch.int, device=device)
    if TORCH_VERSION_AT_LEAST_2_5:
        q = (q[::, ::2] << 4 | q[::, 1::2]).to(torch.uint8)
    packed_w = torch._convert_weight_to_int4pack(q, inner_k_tiles)
    q_groups = k // group_size
    scales = torch.randn(n, q_groups, dtype=torch.bfloat16, device=device)
    zeros = torch.randn_like(scales)
    scales_and_zeros = pack_tinygemm_scales_and_zeros(scales, zeros)

    test_utils = [
    "test_schema",
    "test_autograd_registration",
    "test_faketensor",
    ]
    # TODO: Figure out why test fails unless torch >= 2.5
    if TORCH_VERSION_AT_LEAST_2_5:
        test_utils.append("test_aot_dispatch_dynamic")
    opcheck(
        torch.ops.torchao.dequantize_tensor_core_tiled_layout,
        (packed_w, scales_and_zeros, group_size, inner_k_tiles),
        test_utils=test_utils,
    )


class SparseMarlin24(TestCase):
    TILES = 16

    def _op_check(self, inputs, sparse_w_int4, meta, scales, workspace, thread_k, thread_m, sms=-1, max_par=16):
        out = torch.empty((inputs.size(0), scales.size(1)), dtype=inputs.dtype, device=inputs.device)

        prob_n = inputs.size(0)
        prob_m = out.size(1)
        prob_k = inputs.size(1)
        group_size = -1 if scales.size(0) == 1 else int(prob_k / 2 / scales.size(0))
        device = torch.cuda.current_device()

        test_utils = ["test_schema", "test_autograd_registration", "test_faketensor", "test_aot_dispatch_dynamic"]
        opcheck(
            torch.ops.torchao.marlin_24_mm,
            (
                inputs, sparse_w_int4, meta, out, scales, prob_m, prob_n, prob_k, 
                workspace, group_size, device, thread_k, thread_m, sms, max_par
            ),
            test_utils=test_utils,
        )

    def _gen_values(self, m, n, k, group_size):
        maxq = 2**4 - 1
        inputs = torch.randn((n, k), dtype=torch.half, device="cuda")
        w = torch.randn((m, k), dtype=torch.half, device="cuda")

        w = w.t()
        if group_size != -1:
            w = w.reshape((-1, group_size, m))
            w = w.permute(1, 0, 2)
            w = w.reshape((group_size, -1))

        scales = torch.max(torch.abs(w), 0, keepdim=True)[0]
        scales *= 2 / maxq

        w = torch.round(w / scales).int()
        w += (maxq + 1) // 2
        w = torch.clamp(w, 0, maxq)

        w_fp16 = (w - (maxq + 1) // 2).half() * scales
        scales = scales.reshape((-1, m)).contiguous()

        if group_size != -1:

            def reshape(w):
                w = w.reshape((group_size, -1, m))
                w = w.permute(1, 0, 2)
                w = w.reshape((k, m)).contiguous()
                return w

            w_fp16 = reshape(w_fp16)
            w = reshape(w)
        
        mask = mask_creator(w.T).cuda().bool()
        sparse_w_fp16_ref = (mask * w_fp16.T).T

        return inputs, sparse_w_fp16_ref, w_fp16, scales

    def _run_problem(self, m, n, k, thread_k, thread_m, group_size=-1):
        inputs, sparse_w_fp16_ref, w_fp16, scales = self._gen_values(m, n, k, group_size)
        out_ref = torch.matmul(inputs, sparse_w_fp16_ref)

        # If no groupsize is provided, we assume it is the same as the in_features of the weights
        # https://github.com/IST-DASLab/Sparse-Marlin/blob/c2ffa2395a3ada26c8cb7f910a5ec65bd3ce288a/marlin/__init__.py#L290
        if group_size == -1:
            group_size = k

        w_int4, scales = fp16_to_int4_marlin_format(w_fp16, scales, group_size)
        sparse_w_int4, scales, meta = pack_to_sparse_marlin_24(w_int4, scales, self.TILES)

        workspace = torch.zeros(m // 128 * 16, device="cuda", dtype=torch.int32)
        out = marlin_24_mm(inputs, sparse_w_int4, meta, scales, workspace, thread_k, thread_m, -1)
        torch.cuda.synchronize()

        self.assertLess(
            torch.mean(torch.abs(out - out_ref)) / torch.mean(torch.abs(out_ref)), 0.002
        )

        # TODO(diogo): Enable this check once I understand how to make `out` mutable
        # self._op_check(inputs, sparse_w_int4, meta, scales, workspace, thread_k, thread_m)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_correctness(self):
        self._run_problem(256, 16, 256, 128, 128, -1)
        self._run_problem(21504, 16, 4096, 64, 256, 128)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tiles(self):
        for m in [1, 2, 4, 8, 12, 16, 32, 64]:
            for thread_k, thread_n in [(64, 256), (128, 128)]:
                if m > 16 and thread_k == 128:
                    continue
                self._run_problem(2 * 256, m, 1024, thread_k, thread_n)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_k_stages_divisibility(self):
        for k in [3 * 64 + 64 * 4 * 2 + 64 * i for i in range(1, 4)]:
            self._run_problem(2 * 256, 16, k, 64, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_very_few_stages(self):
        for k in [64, 128, 192]:
            self._run_problem(3 * 256, 16, k, 64, 256)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_llama_shapes(self):
        MODELS = {
            " 7B": [(4096, 3 * 4096), (4096, 4096), (4096, 2 * 10752), (10752, 4096)],
            "13B": [(5120, 3 * 5120), (5120, 5120), (5120, 2 * 13568), (13568, 5120)],
            "33B": [(6656, 3 * 6656), (6656, 6656), (6656, 2 * 17664), (17664, 6656)],
            "70B": [(8192, 3 * 8192), (8192, 8192), (8192, 2 * 21760), (21760, 8192)],
        }

        try:
            for _, layers in MODELS.items():
                for layer in layers:
                    for thread_k, thread_m in [(128, 128)]:
                        for batch in [16]:
                            print(layer[1], batch, layer[0])
                            self._run_problem(layer[1], batch, layer[0], thread_k, thread_m)
        # If someone runs this on a GPU with less than 24GB of memory, it will run out of memory
        # but we don't want to fail the test
        except torch.OutOfMemoryError:
            pass

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_groups(self):
        for m in [16]:
            for groupsize in [128]:
                for n, k in [(256, 512), (256, 1024), (256 * 128, 1024)]:
                    for thread_shape in [(128, 128), (64, 256)]:
                        self._run_problem(n, m, k, *thread_shape, groupsize)


if __name__ == "__main__":
    run_tests()
