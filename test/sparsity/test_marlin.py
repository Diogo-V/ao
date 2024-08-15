import torch
import copy
import pytest

from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests
from torchao.utils import TORCH_VERSION_AFTER_2_3
from torchao.dtypes import MarlinSparseLayoutType
from torchao.sparsity.sparse_api import apply_fake_sparsity
from torchao.quantization.quant_api import int4_weight_only, quantize_


class TestQuantSparseMarlin(TestCase):

    @pytest.mark.skipif(not TORCH_VERSION_AFTER_2_3, reason="pytorch 2.3+ feature")
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="Need CUDA available")
    def test_quant_sparse_marlin_layout(self):
        input = torch.rand((128, 128)).half().cuda()
        model = (
            nn.Sequential(
                nn.Linear(128, 256),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 128),
            )
            .half()
            .cuda()
        )

        apply_fake_sparsity(model)
        model_copy = copy.deepcopy(model)
        model_copy = model_copy.bfloat16()

        quantize_(model_copy, int4_weight_only())
        dense_result = model_copy(input.bfloat16()).to(torch.float16)

        quantize_(model, int4_weight_only(layout_type=MarlinSparseLayoutType()))
        sparse_result = model(input)

        assert torch.allclose(dense_result, sparse_result, rtol=1e-2, atol=1e-2)      

    # TODO(diogo): Add rest of tests from sparse marlin repo


if __name__ == "__main__":
    run_tests()
