import torch
import copy
import pytest

from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests
from torchao.dtypes import MarlinSparseLayoutType
from torchao.sparsity.sparse_api import apply_fake_sparsity
from torchao.quantization.quant_api import int4_weight_only, quantize_
from torchao.sparsity.marlin import (
    pack_to_marlin_24,
    unpack_from_marlin_24,
    inject_24
)


class SparseMarlin24(TestCase):

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_quant_sparse_marlin_layout_e2e(self):
        input = torch.randn((16, 4096), dtype=torch.float16, device="cuda")
        model = (
            nn.Sequential(
                nn.Linear(4096, 21504),
                nn.Linear(21504, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
                nn.ReLU(),
                nn.Linear(128, 4096),
            )
            .half()
            .cuda()
        )

        # Baseline
        ref_result = model(input)

        apply_fake_sparsity(model)
        model_copy = copy.deepcopy(model)

        # Quantized
        quantize_(model_copy.bfloat16(), int4_weight_only())
        dense_result = model_copy(input.bfloat16()).half()

        # Sparse + quantized
        quantize_(model, int4_weight_only(layout_type=MarlinSparseLayoutType()))
        sparse_result = model(input)

        error_dense = torch.mean(torch.abs(ref_result - dense_result) ** 2)
        error_sparse = torch.mean(torch.abs(ref_result - sparse_result) ** 2)
        assert torch.allclose(error_dense, error_sparse, atol=1e-3), "Mean error is not close"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_pack_unpack_equivalence(self):
        num_bits = 4
        group_size = 128
        shape = (512, 4096)
        w_q = torch.randint(0, 15, shape).int().cuda()
        scales = torch.rand(4096).cuda()

        w_q_24, _ = inject_24(w_q, *w_q.shape)

        # Test pack/unpack equivalence
        q_w_comp, packed_scales, meta = pack_to_marlin_24(w_q_24, scales, num_bits, group_size)
        unpacked_q_w, unpacked_scales = unpack_from_marlin_24(
            q_w_comp, packed_scales, meta, shape, group_size, num_bits
        )

        assert torch.equal(w_q, unpacked_q_w), "Unpacked weights do not match original weights"
        assert torch.equal(scales, unpacked_scales), "Unpacked scales do not match original scales"


if __name__ == "__main__":
    run_tests()
