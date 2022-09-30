import torch
import torch.nn as nn

class DWSConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel=3, padding=1, kernels_per_layer=1):
        super(DWSConv, self).__init__()
        self.out_channel = out_channel
        self.DWConv = nn.Conv2d(in_channel, in_channel * kernels_per_layer, kernel_size=kernel, padding=padding,
                                groups=in_channel, bias=False)
        self.bn = nn.BatchNorm2d(in_channel * kernels_per_layer)
        self.selu = nn.SELU()
        self.PWConv = nn.Conv2d(in_channel * kernels_per_layer, out_channel, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        x = self.DWConv(x)
        x = self.selu(self.bn(x))
        out = self.PWConv(x)
        out = self.selu(self.bn2(out))

        return out


class Fuse(nn.Module):
    def __init__(self, input_channels, output_channels) -> None:
        super(Fuse, self).__init__()

        self.frames_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )

        self.opticals_conv = nn.Sequential(
            nn.Conv2d(in_channels=input_channels, out_channels=output_channels, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid()
        )

        self.fuse_conv = nn.Sequential(
            DWSConv(input_channels//2, output_channels), 
            nn.ReLU(inplace=True)
        )

    def forward(self, frames, opticals):
        frames = self.frames_conv(frames)
        opticals = self.opticals_conv(opticals)
        #fuse = self.fuse_conv(frames * opticals)
        fuse = frames * opticals
        return fuse


def build_fuse(cfg):
    input_channels = cfg.MODEL.FUSE.INPUT_CHANNELS
    output_channels = cfg.MODEL.FUSE.OUTPUT_CHANNELS
    return Fuse(input_channels, output_channels)

if __name__ == "__main__":
    frames = torch.rand((1, 128, 8, 8))
    opticals = torch.rand((1, 128, 8, 8))
    fuse = Fuse(128, 16)
    fused = fuse(frames, opticals)
    print(fused.shape)