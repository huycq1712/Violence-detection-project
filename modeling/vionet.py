import torch
import torch.nn as nn
import torchsummary
import torch.nn.functional as F
from .backbone.resnet import build_resnet, ResNet, model_archs
from .backbone.sepconvlstm import build_sepconvlstm, SepConvLSTM
from .fuse.fuse import build_fuse, Fuse
from .fcn.fcn import build_seg_head, SegHead


class Attn(nn.Module):
    def __init__(self, h_dim):
        super(Attn, self).__init__()
        self.h_dim = h_dim
        self.main = nn.Sequential(
            nn.Linear(h_dim, 24),
            nn.ReLU(True),
            nn.Linear(24,1)
        )

    def forward(self, encoder_outputs):
        b_size = encoder_outputs.size(0)
        attn_ene = self.main(encoder_outputs.view(-1, self.h_dim)) # (b, s, h) -> (b * s, 1)
        return F.softmax(attn_ene.view(b_size, -1), dim=1).unsqueeze(2) # (b*s, 1) -> (b, s, 1)

class VioNet(nn.Module):
    def __init__(self, cfg=None) -> None:
        super(VioNet, self).__init__()

        self.frames_encoder = build_resnet(cfg)
        self.opticals_encoder = build_resnet(cfg)
        self.frames_temproral_encoder = build_sepconvlstm(cfg)
        self.opticals_temproral_encoder = build_sepconvlstm(cfg)


        self.segments = build_seg_head(cfg)
        self.fuse = build_fuse(cfg)

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(cfg.MODEL.FUSE.OUTPUT_CHANNELS, cfg.MODEL.FUSE.OUTPUT_CHANNELS//2),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.MODEL.FUSE.OUTPUT_CHANNELS//2, cfg.MODEL.NUM_CLASSES),
        )

        """self.classifier = nn.Sequential(
            #nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(2048, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2)
        )"""
    
    def forward(self, frames, opticals, batch_first=True, is_train=True):
        
        frames = frames.permute(1, 0, 2, 3, 4) if batch_first else frames
        opticals = opticals.permute(1, 0, 2, 3, 4) if batch_first else opticals
        
        frames_reshape = frames.contiguous().view(-1, *frames.shape[2:])
        opticals_reshape = opticals.contiguous().view(-1, *opticals.shape[2:])
        segs_feat, frames_feat = self.frames_encoder(frames_reshape)

        if is_train:
            #print(segs_pred.shape)
            segs_pred = self.segments(segs_feat)
            
        opticals_feat = self.opticals_encoder(opticals_reshape)[1]


        #print(frames_feat.shape)

        frames_feat = frames_feat.view(frames.shape[0], frames.shape[1], *frames_feat.shape[1:])
        opticals_feat = opticals_feat.view(opticals.shape[0], opticals.shape[1], *opticals_feat.shape[1:])
        #segs_pred = segs_pred.view(frames.shape[0], frames.shape[1], *segs_pred.shape[1:]) if is_train else None

        #print(frames_feat.shape)

        frames_feat = self.frames_temproral_encoder(frames_feat).max(dim=1)[0]
        opticals_feat = self.opticals_temproral_encoder(opticals_feat).max(dim=1)[0]

        fuse_feat = self.fuse(frames_feat, opticals_feat)
        output =self.classifier(fuse_feat)

        #print(output.shape)

        return output, segs_pred if is_train else output, None

def build_vionet(cfg):
    
    model = VioNet(cfg)
    return model

if __name__ == "__main__":
    import time
    
    
    
    frames = torch.rand((1, 24, 3, 244, 244)).to('cuda')
    opticals = torch.rand((1, 24, 3, 244, 244)).to('cuda')

    vionet = VioNet().to('cuda')
    start_time = time.time()
    output = vionet(frames, opticals)
    print(output[0].shape)
    print("--- %s seconds ---" % (time.time() - start_time))

    #print(vionet)
    #torchsummary.summary(vionet, [(24, 3, 244, 244), (24, 3, 244, 244)])

    