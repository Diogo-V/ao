import torch
from torch.ao.quantization.observer import UniformQuantizationObserverBase

__all__ = ["PerChannelNormObserver", "mask_creator"]


# Observers
class PerChannelNormObserver(UniformQuantizationObserverBase):
    """
    A custom observer that computes the L2 norm of each channel and stores it in a buffer.
    """

    def __init__(self, **kwargs) -> None:
        # init with fixed qparams for quantization flow
        super().__init__(
            dtype=torch.quint8,
            qscheme=torch.per_channel_affine,
            reduce_range=False,
            quant_min=None,
            quant_max=None,
            eps=torch.finfo(torch.float32).eps,
            **kwargs
        )
        # set averaging constant so quantization flow knows observer is memoryless.
        self.averaging_constant = 1.0
        self.register_buffer("norm", torch.tensor([]))

    #  inconsistently.

    def forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape

        # channel_ax is always the last dimension
        new_axis_list = [i for i in range(x.dim())]  # noqa: C416
        new_axis_list[0], new_axis_list[-1] = new_axis_list[-1], new_axis_list[0]
        y = x.permute(new_axis_list)
        y = torch.flatten(y, start_dim=1)
        norm = torch.norm(y, dim=1) ** 2

        if self.norm.numel() == 0:
            self.norm.resize_(norm.shape)
            self.norm.copy_(norm)
        else:
            self.norm += norm

        return x_orig

    #  inconsistently.

    def calculate_qparams(self):
        raise NotImplementedError(
            "PerChannelNormObserver is designed to store activations only. "
        )


def mask_creator(
        tensor: torch.Tensor,
        N: int = 2,
        M: int = 4,
    ) -> torch.Tensor:
    """
    Class for creating N:M sparsity masks.
    Masks will be created using the N:M ratio, where for every block of 
    M weights, N will be pruned based on ranked weight value. Each mask 
    will correspond to the given tensor.

    :param tensor: The input tensor to create a mask for
    :param N: The number of weights in a group to keep
    :param M: The size of a weight group

    :return: A mask tensor with the same shape as the input tensor
    """
    N = 2
    M = 4

    mask = None
    # for i, tensor in enumerate(tensors):
    if tensor.numel() % M != 0:
        raise ValueError(
            f"Tensor of size {tensor.shape} can't be evenly divided into "
            f"{M} groups")

    num_groups = tensor.numel() // M

    # N:M sparsity for linear layers
    tensor_temp = tensor.detach().abs().reshape(num_groups, M)
    index = torch.argsort(tensor_temp, dim=1)[:, :int(M - N)]

    w_b = torch.ones(tensor_temp.shape, device=tensor_temp.device)
    mask = w_b.scatter_(dim=1, index=index, value=0).reshape(tensor.shape)

    return mask
