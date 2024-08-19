import torch
import copy
import pytest

from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests
from torchao.utils import compute_max_diff
from torchao.dtypes import MarlinSparseLayoutType
from torchao.sparsity.sparse_api import apply_fake_sparsity
from torchao.quantization.quant_api import int4_weight_only, quantize_
from torchao.sparsity.marlin_utils import (
    pack_to_marlin_24,
    unpack_from_marlin_24,
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

        apply_fake_sparsity(model)
        model_copy = copy.deepcopy(model)

        # Baseline to match against
        quantize_(model_copy.bfloat16(), int4_weight_only())
        dense_result = model_copy(input.bfloat16()).half()

        # Sparse + quantized
        quantize_(model, int4_weight_only(layout_type=MarlinSparseLayoutType()))
        sparse_result = model(input)

        max_diff = compute_max_diff(dense_result, sparse_result)
        assert max_diff < 0.50, f"Max diff: {max_diff}"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_pack_unpack_equivalence(self):
        tiles = 16
        num_bits = 4
        shape = (512, 4096)
        w_int4 = torch.randint(0, 15, shape).int().cuda()
        scales = torch.rand(4096).cuda()

        # Test pack/unpack equivalence
        sparse_w_int4, packed_scales, meta = pack_to_marlin_24(w_int4, scales, tiles, num_bits)
        unpacked_w_int4, unpacked_scales = unpack_from_marlin_24(sparse_w_int4, packed_scales, meta, tiles, shape)

        # When unpacking, that values that were masked will be zeroed out. So, we need
        # to zero out the same values in the original weights to compare
        makeshift_mask = unpacked_w_int4 == 0
        w_int4[makeshift_mask] = 0

        assert torch.equal(w_int4, unpacked_w_int4), "Unpacked weights do not match original weights"
        assert torch.equal(scales, unpacked_scales), "Unpacked scales do not match original scales"


if __name__ == "__main__":
    run_tests()
