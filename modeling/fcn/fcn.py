from pyexpat import model
import torch
import torch.nn as nn


class SegHead(nn.Module):
    """
    Segmentation head for multitask learning
    """
    def __init__(self, input_channels) -> None:
        super(SegHead, self).__init__()

        self.input_channels = input_channels

        seg_head = []

        # Encoder
        seg_head.append(self._make_conv_layer(self.input_channels, self.input_channels//2, 3, 2, upsample=False))
        seg_head.append(self._make_conv_layer(self.input_channels//2, self.input_channels//4, 3, 2, upsample=False))
        #seg_head.append(self._make_conv_layer(self.input_channels//4, self.input_channels//8, 3, 2, upsample=False))

        # Decoder
        seg_head.append(self._make_conv_layer(self.input_channels//4, self.input_channels//8, 3, 2, upsample=True))
        seg_head.append(self._make_conv_layer(self.input_channels//8, self.input_channels//16, 3, 2, upsample=True))
        seg_head.append(self._make_conv_layer(self.input_channels//16, self.input_channels//32, 3, 2, upsample=True))
        seg_head.append(self._make_conv_layer(self.input_channels//32, self.input_channels//32, 3, 2, upsample=True))
        seg_head.append(self._make_conv_layer(self.input_channels//32, self.input_channels//64, 3, 2, upsample=True))
        #seg_head.append(self._make_conv_layer(self.input_channels//4, self.input_channels//8, 3, 2, upsample=True))
        #seg_head.append(self._make_conv_layer(self.input_channels//8, self.input_channels//16, 3, 2, upsample=True))
        #seg_head.append(self._make_conv_layer(self.input_channels//16, 1, 1, 1, upsample=False))

        seg_head.append(nn.Sequential(
            nn.Conv2d(self.input_channels//64, 1, kernel_size=1, stride=1, padding=0), 
            nn.Sigmoid()
        ))

        # Segmentation head
        self.seg_head = nn.Sequential(*seg_head)


    def forward(self, x):
        return self.seg_head(x)


    def _make_conv_layer(self, in_channels, out_channels, kernel, stride, padding=1, upsample=False):
        """
        Make a convolutional layer
        """
        if upsample:
            return nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
                #nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1),
                #nn.ReLU(inplace=True),
                #nn.BatchNorm2d(out_channels),
            )

        return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=kernel, stride=stride, padding=padding),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(out_channels),
            )


def build_seg_head(cfg):
    input_channels = cfg.MODEL.SEG_HEAD.INPUT_CHANNELS
    return SegHead(input_channels)

if __name__ == "__main__":

    model = SegHead(input_channels=128)
    #[1, 512, 8, 8])
    img = torch.randn(9, 128, 31, 31)

    print(model(img).shape)