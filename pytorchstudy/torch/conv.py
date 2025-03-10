import torch
import torch.nn as nn


def conv1d(
    in_chan: int = 1, out_chan: int = 100, kernel_size: int = 4, padding: int = 0
):
    # Create a Conv1d layer
    conv_layer = nn.Conv1d(
        in_channels=in_chan,
        out_channels=out_chan,
        kernel_size=kernel_size,
        stride=kernel_size // 2,
        padding=padding,
    )

    print(f"conv weight shape: {conv_layer.weight.shape}")
    print(f"conv weight:\n{conv_layer.weight}")

    # Generate random input tensor
    batch_size = 2
    sequence_length = 200
    input_tensor = torch.randn(batch_size, 1, sequence_length)
    print(f"input tensor shape: {input_tensor.shape}")

    # Apply the convolutional layer to the input tensor
    output_tensor = conv_layer(input_tensor)
    print(f"output tensor shape: {output_tensor.shape}")
    return output_tensor


def deconv1d(
    y, w, in_chan: int, out_chan: int, ks: int, stride: int, padding: int, bias: bool
):
    """_summary_
    How PyTorch Transposed Convs1D Work
    https://medium.com/@santi.pdp/how-pytorch-transposed-convs1d-work-a7adac63c4a5

    Args:
        y (_type_): _description_
        w (_type_): _description_
        in_chan (int): _description_
        out_chan (int): _description_
        ks (int): _description_
        stride (int): _description_
        padding (int): _description_
        bias (bool): _description_

    Returns:
        _type_: _description_
    """
    deconv = nn.ConvTranspose1d(
        in_channels=in_chan,
        out_channels=out_chan,
        kernel_size=ks,
        stride=stride,
        padding=padding,
        bias=bias,
    )
    deconv.weight.data = w
    return deconv(y)
