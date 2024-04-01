import startai


class UNetDoubleConv(startai.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.mid_channels = mid_channels if mid_channels else out_channels
        super(UNetDoubleConv, self).__init__()

    def _build(self, *args, **kwargs):
        self.double_conv = startai.Sequential(
            startai.Conv2D(
                self.in_channels, self.mid_channels, [3, 3], 1, 1, with_bias=False
            ),
            startai.BatchNorm2D(self.mid_channels, training=False),
            startai.ReLU(),
            startai.Conv2D(
                self.mid_channels, self.out_channels, [3, 3], 1, 1, with_bias=False
            ),
            startai.BatchNorm2D(self.out_channels, training=False),
            startai.ReLU(),
        )

    def _forward(self, x):
        return self.double_conv(x)


class UNetDown(startai.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super().__init__()

    def _build(self, *args, **kwargs):
        self.maxpool_conv = startai.Sequential(
            startai.MaxPool2D(2, 2, 0), UNetDoubleConv(self.in_channels, self.out_channels)
        )

    def _forward(self, x):
        return self.maxpool_conv(x)


class UNetUp(startai.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        super().__init__()

    def _build(self, *args, **kwargs):
        if self.bilinear:
            self.up = startai.interpolate(
                scale_factor=2, mode="bilinear", align_corners=True
            )
            self.conv = UNetDoubleConv(
                self.in_channels, self.out_channels, self.in_channels // 2
            )
        else:
            self.up = startai.Conv2DTranspose(
                self.in_channels, self.in_channels // 2, [2, 2], 2, "VALID"
            )
            self.conv = UNetDoubleConv(self.in_channels, self.out_channels)

    def _forward(self, x1, x2):
        x1 = self.up(x1)
        # input is BHWC
        diff_H = x2.shape[1] - x1.shape[1]
        diff_W = x2.shape[2] - x1.shape[2]

        pad_width = (
            (0, 0),
            (diff_H - diff_H // 2, diff_H // 2),
            (diff_W // 2, diff_W - diff_W // 2),
            (0, 0),
        )

        x1 = startai.constant_pad(x1, pad_width)
        x = startai.concat((x2, x1), axis=3)
        return self.conv(x)


class UNetOutConv(startai.Module):
    def __init__(self, in_channels, out_channels):
        self.in_channels = in_channels
        self.out_channels = out_channels
        super(UNetOutConv, self).__init__()

    def _build(self, *args, **kwargs):
        self.conv = startai.Conv2D(self.in_channels, self.out_channels, [1, 1], 1, 0)

    def _forward(self, x):
        return self.conv(x)
