from typing import Callable, Optional, Union, Tuple
import startai


def _make_divisible(v: float, divisor: int, min_value: Optional[int] = None) -> int:
    """
    Taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)


class EfficientNetConv2dNormActivation(startai.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, ...]] = 3,
        stride: Union[int, Tuple[int, ...]] = 1,
        padding: Optional[Union[int, Tuple[int, ...], str]] = None,
        groups: int = 1,
        norm_layer: Optional[Callable[..., startai.Module]] = startai.BatchNorm2D,
        activation_layer: Optional[Callable[..., startai.Module]] = startai.ReLU,
        dilation: Union[int, Tuple[int, ...]] = 1,
        inplace: Optional[bool] = True,
        bias: Optional[bool] = None,
        conv_layer: Callable[..., startai.Module] = startai.Conv2D,
        depthwise: bool = False,
    ) -> None:
        if padding is None:
            padding = (kernel_size - 1) // 2 * dilation
        if bias is None:
            bias = norm_layer is None
        if not depthwise:
            layers = [
                conv_layer(
                    in_channels,
                    out_channels,
                    [kernel_size, kernel_size],
                    stride,
                    padding,
                    dilations=dilation,
                    with_bias=bias,
                )
            ]
        else:
            layers = [
                startai.DepthwiseConv2D(
                    in_channels,
                    [kernel_size, kernel_size],
                    stride,
                    padding,
                    dilations=dilation,
                    with_bias=bias,
                )
            ]

        if norm_layer is not None:
            layers.append(norm_layer(out_channels, training=False))

        if activation_layer is not None:
            layers.append(activation_layer())
        super().__init__(*layers)
        self.out_channels = out_channels


class EfficientNetSqueezeExcitation(startai.Module):
    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation: Callable[..., startai.Module] = startai.ReLU,
        scale_activation: Callable[..., startai.Module] = startai.sigmoid,
    ) -> None:
        self.fc1 = startai.Conv2D(input_channels, squeeze_channels, [1, 1], 1, 0)
        self.fc2 = startai.Conv2D(squeeze_channels, input_channels, [1, 1], 1, 0)
        self.activation = activation()
        self.scale_activation = scale_activation

        super().__init__()

    def _scale(self, x: startai.Array) -> startai.Array:
        x = startai.permute_dims(x, (0, 3, 1, 2))
        x = startai.adaptive_avg_pool2d(x, 1)
        scale = startai.permute_dims(x, (0, 2, 3, 1))
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def _forward(self, input: startai.Array) -> startai.Array:
        scale = self._scale(input)
        return scale * input


def stochastic_depth(x, p):
    """
    Implements the Stochastic Depth from `"Deep Networks with Stochastic Depth"
    <https://arxiv.org/abs/1603.09382>`_ used for randomly dropping residual
    branches of residual architectures.

    Args:
        input (Tensor[N, ...]): The input tensor or arbitrary dimensions with
                    the first one being its batch i.e. a batch with ``N`` rows.
        p (float): probability of the input to be zeroed.

    Returns
    -------
        Tensor[N, ...]: The randomly zeroed tensor.
    """
    survival_rate = 1.0 - p
    binary_tensor = (
        startai.random_uniform(shape=(x.shape[0], 1, 1, 1), low=0, high=1, device=x.device)
        < survival_rate
    )
    return startai.divide(x, survival_rate) * binary_tensor


class EfficientNetStochasticDepth(startai.Module):
    """See :func:`stochastic_depth`."""

    def __init__(self, p: float) -> None:
        self.p = p
        super().__init__()

    def _forward(self, input: startai.Array) -> startai.Array:
        return stochastic_depth(input, self.p)
