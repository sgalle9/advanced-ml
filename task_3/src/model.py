import torch
from torch import nn


class ConvBlock(nn.Module):
    """
    A convolutional block consisting of two convolutional layers, each followed by batch normalization and ReLU activation.
    """

    def __init__(self, in_channels, out_channels, mid_channels=None, kernel_size=3):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels

        # First convolutional block.
        self.conv1 = nn.Conv2d(
            in_channels, mid_channels, kernel_size=kernel_size, padding="same"
        )
        self.batch_norm1 = nn.BatchNorm2d(mid_channels)

        # Second convolutional block.
        self.conv2 = nn.Conv2d(
            mid_channels, out_channels, kernel_size=kernel_size, padding="same"
        )
        self.batch_norm2 = nn.BatchNorm2d(out_channels)

        # Activation function
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)

        return x


class UNet(nn.Module):
    """
    U-Net architecture consisting of an encoder and decoder.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        mid_channels,
        kernel_size,
        max_pool_kernel_size,
        up_kernel_size,
        up_stride,
        dropout_p,
    ):
        super().__init__()

        self.dropout = nn.Dropout2d(p=dropout_p)

        ##################################################
        # Encoder
        ##################################################
        self.encoders = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=in_channels,
                    out_channels=mid_channels[0],
                    kernel_size=kernel_size,
                )
            ]
        )
        self.encoders.extend(
            [
                ConvBlock(
                    in_channels=mid_channels[i],
                    out_channels=mid_channels[i + 1],
                    kernel_size=kernel_size,
                )
                for i in range(len(mid_channels) - 2)
            ]
        )

        self.bottleneck_layer = ConvBlock(
            in_channels=mid_channels[-2],
            out_channels=mid_channels[-1],
            kernel_size=kernel_size,
        )

        self.max_pool = torch.nn.MaxPool2d(kernel_size=max_pool_kernel_size)

        ##################################################
        # Decoder
        ##################################################
        self.decoders = nn.ModuleList(
            [
                ConvBlock(
                    in_channels=mid_channels[i],
                    out_channels=mid_channels[i - 1],
                    kernel_size=kernel_size,
                )
                for i in reversed(range(1, len(mid_channels)))
            ]
        )

        self.upconvs = nn.ModuleList(
            [
                nn.ConvTranspose2d(
                    in_channels=mid_channels[i],
                    out_channels=mid_channels[i - 1],
                    kernel_size=up_kernel_size,
                    stride=up_stride,
                )
                for i in reversed(range(1, len(mid_channels)))
            ]
        )

        # Final convolution to get the segmentation map.
        self.final_conv = nn.Conv2d(mid_channels[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.max_pool(x)

        x = self.bottleneck_layer(x)

        x = self.dropout(x)

        skips = reversed(skips)
        for decoder, upconv, skip in zip(self.decoders, self.upconvs, skips):
            x = upconv(x)
            x = torch.cat((x, skip), dim=1)  # Along `Channels` dimension.
            x = decoder(x)

        x = self.dropout(x)

        return self.final_conv(x)

    def predict(self, x):
        """
        Predict the segmentation map from input images.
        :param x: Input tensor to be segmented.
        :return: Predicted segmentation map as a tensor of class indices.
        """
        y_hat = self(x).argmax(axis=1)
        return y_hat
