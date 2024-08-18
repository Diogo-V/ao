import torch
import copy
import pytest

from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests
from torchao.dtypes import MarlinSparseLayoutType
from torchao.sparsity.sparse_api import apply_fake_sparsity
from torchao.quantization.quant_api import int4_weight_only, quantize_
from torchao.sparsity.marlin import (
    pack_to_sparse_marlin_24,
    unpack_from_sparse_marlin_24,
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
            )
            .half()
            .cuda()
        )

        apply_fake_sparsity(model)
        model_copy = copy.deepcopy(model)

        # Baseline to match against
        quantize_(model_copy.bfloat16(), int4_weight_only())
        dense_result = model_copy(input.bfloat16()).half()

        # Sparse + quantized
        quantize_(model, int4_weight_only(layout_type=MarlinSparseLayoutType()))
        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-2, atol=1e-2), "Sparse and dense results do not match"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_pack_unpack_equivalence(self):
        tiles = 16
        shape = (512, 4096)
        w_int4 = torch.randint(0, 15, shape).int().cuda()
        scales = torch.rand(4096).cuda()

        # Test pack/unpack equivalence
        sparse_w_int4, packed_scales, meta = pack_to_sparse_marlin_24(w_int4, scales, tiles)
        unpacked_w_int4, unpacked_scales = unpack_from_sparse_marlin_24(sparse_w_int4, packed_scales, meta, tiles, shape)

        # When unpacking, that values that were masked will be zeroed out. So, we need
        # to zero out the same values in the original weights to compare
        makeshift_mask = unpacked_w_int4 == 0
        w_int4[makeshift_mask] = 0

        assert torch.equal(w_int4, unpacked_w_int4), "Unpacked weights do not match original weights"
        assert torch.equal(scales, unpacked_scales), "Unpacked scales do not match original scales"


if __name__ == "__main__":
    run_tests()
