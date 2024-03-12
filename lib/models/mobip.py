import torch
from torch import tensor
import torch.nn as nn
import sys,os
import math
import sys
import time
from lib.core.function import AverageMeter
sys.path.append(os.getcwd())
from lib.utils import initialize_weights
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv, Seg_Feature_Fusion, SegmentationHead, Temp_Conv
from torch.nn import Upsample
from lib.utils import check_anchor_order
from lib.core.evaluate import SegmentationMetric
from lib.utils.utils import time_synchronized
from .MobileNetV2_component import Wrapped_InvertedResidual


SEG_DEPTH = 1

Model = [
[24, 31 ],   #Det_out_idx, Segout_idx
[ -1, Focus, [3, 16, 3]],   #0 [16, 320, 320]
#                                inp, out, t, n, s
[-1, Wrapped_InvertedResidual, [16, 16, 1, 1, 1]], #1 [320, 320]
[-1, Wrapped_InvertedResidual, [16, 24, 6, 2, 2]], #2 [160, 160]
[-1, Wrapped_InvertedResidual, [24, 32, 6, 3, 2]], #3 [80, 80]
[-1, Wrapped_InvertedResidual, [32, 64, 6, 4, 2]], #4 [40, 40]
[-1, Wrapped_InvertedResidual, [64, 96, 6, 3, 1]], #5 [40, 40]
[-1, Wrapped_InvertedResidual, [96, 160, 6, 3, 2]], #6 [20, 20]
[-1, Wrapped_InvertedResidual, [160, 320, 6, 1, 1]], #7 [20, 20]
[ -1, SPP, [320, 320, [5, 9, 13]]], #8   20x20

[-1, Wrapped_InvertedResidual, [320, 320, 6, 2, 1]], #9   20x20
[-1, Wrapped_InvertedResidual, [320, 96, 6, 1, 1]], #10   20x20
[ -1, Upsample, [None, 2, 'nearest']], #11 40x40
[ [-1, 5], Concat, [1]], #12 40x40
[-1, Wrapped_InvertedResidual, [192, 64, 6, 2, 1, True]], #13 40x40
[-1, Wrapped_InvertedResidual, [64, 32, 6, 1, 1, True]], #14 40x40
[ -1, Upsample, [None, 2, 'nearest']],  #15 80x80
[ [-1,3], Concat, [1]], #16

[-1, Wrapped_InvertedResidual, [64, 64, 6, 2, 1, True]], #17
[-1, Wrapped_InvertedResidual, [64, 64, 6, 1, 2, True]], #18
[ [-1, 13], Concat, [1]],       #19
[-1, Wrapped_InvertedResidual, [128, 128, 6, 2, 1, True]], #20
[-1, Wrapped_InvertedResidual, [128, 96, 6, 1, 2, True]], #21
[ [-1, 10], Concat, [1]],   #22
[-1, Wrapped_InvertedResidual, [192, 192, 6, 2, 1, True]], #23
[ [17, 20, 23], Detect,  [1, [[3,9,5,11,4,20], [7,18,6,39,12,31], [19,50,38,81,68,157]], [64, 128, 192]]], #Detection head  24

[ 16, Conv, [64, 32, 1, 1]],   #25   [80, 80, 80]
[ -1, Upsample, [None, 2, 'bilinear']],  #26  160x160
[-1, Wrapped_InvertedResidual, [32, 24, 6, SEG_DEPTH, 1, False]], #27
[ -1, Upsample, [None, 2, 'bilinear']],  #28  320x320
[-1, Wrapped_InvertedResidual, [24, 16, 6, SEG_DEPTH, 1, False]], #29
[ -1, Upsample, [None, 2, 'bilinear']],  #30  640x640
[ -1, Conv, [16, 3, 1, 1]], #31 Segmentation head
]


class MCnet(nn.Module):
    def __init__(self, block_cfg, **kwargs):
        super(MCnet, self).__init__()
        layers, save= [], []
        self.nc = 1
        self.detector_index = -1
        self.det_out_idx = block_cfg[0][0]
        self.seg_out_idx = block_cfg[0][1:]
        self.consume_time_list = []
        self.backbone_time = AverageMeter()
        self.neck_time = AverageMeter()
        self.det_time = AverageMeter()
        self.seg_time = AverageMeter()

        # Build model
        for i, (from_, block, args) in enumerate(block_cfg[1:]):
            block = eval(block) if isinstance(block, str) else block  # eval strings
            if block is Detect:
                self.detector_index = i
            block_ = block(*args)
            block_.index, block_.from_ = i, from_
            layers.append(block_)
            save.extend(x % i for x in ([from_] if isinstance(from_, int) else from_) if x != -1)  # append to savelist
        assert self.detector_index == block_cfg[0][0]

        self.model, self.save = nn.Sequential(*layers), sorted(save)
        self.names = [str(i) for i in range(self.nc)]

        # set stride、anchor for detector
        Detector = self.model[self.detector_index]  # detector
        if isinstance(Detector, Detect):
            s = 128  # 2x min stride
            with torch.no_grad():
                model_out = self.forward(torch.zeros(1, 3, s, s))
                detects, _= model_out
                Detector.stride = torch.tensor([s / x.shape[-2] for x in detects])  # forward
            Detector.anchors /= Detector.stride.view(-1, 1, 1)  # Set the anchors for the corresponding scale
            check_anchor_order(Detector)
            self.stride = Detector.stride
            self._initialize_biases()
        
        initialize_weights(self)

    def forward(self, x):
        cache = []
        out = []
        det_out = None
        for i, block in enumerate(self.model):
            if block.from_ != -1:
                x = cache[block.from_] if isinstance(block.from_, int) else [x if j == -1 else cache[j] for j in block.from_]       #calculate concat detect
            x = block(x)
            if i in self.seg_out_idx:     #save driving area segment result
                # m=nn.Sigmoid()
                m=nn.Identity()
                out.append(m(x))
            if i == self.detector_index:
                det_out = x
            cache.append(x if block.index in self.save else None)
        out.insert(0,det_out)
        return out

    def _initialize_biases(self, cf=None):  # initialize biases into Detect(), cf is class frequency
        # https://arxiv.org/abs/1708.02002 section 3.3
        # cf = torch.bincount(torch.tensor(np.concatenate(dataset.labels, 0)[:, 0]).long(), minlength=nc) + 1.
        # m = self.model[-1]  # Detect() module
        m = self.model[self.detector_index]  # Detect() module
        for mi, s in zip(m.m, m.stride):  # from
            b = mi.bias.view(m.na, -1)  # conv.bias(255) to (3,85)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)  # obj (8 objects per 640 image)
            b.data[:, 5:] += math.log(0.6 / (m.nc - 0.99)) if cf is None else torch.log(cf / cf.sum())  # cls
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

def get_net(cfg, **kwargs): 
    m_block_cfg = Model
    model = MCnet(m_block_cfg, **kwargs)
    return model


if __name__ == "__main__":
    from torch.utils.tensorboard import SummaryWriter
    model = get_net(False)
    input_ = torch.randn((1, 3, 256, 256))
    gt_ = torch.rand((1, 2, 256, 256))
    metric = SegmentationMetric(2)
    model_out,SAD_out = model(input_)
    detects, dring_area_seg, lane_line_seg = model_out
    Da_fmap, LL_fmap = SAD_out
    for det in detects:
        print(det.shape)
    print(dring_area_seg.shape)
    print(lane_line_seg.shape)
 
