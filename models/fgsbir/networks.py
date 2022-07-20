import torch.nn as nn
import torchvision.models as backbone_
import torch.nn.functional as F
import torch



class Resnet50_Network(nn.Module):
    def __init__(self):
        super(Resnet50_Network, self).__init__()
        backbone = backbone_.resnet50(weights=backbone_.ResNet50_Weights.DEFAULT) #resnet50, resnet18, resnet34

        self.features = nn.Sequential()
        for name, module in backbone.named_children():
            if name not in ['avgpool', 'fc']:
                self.features.add_module(name, module)
        self.pool_method =  nn.AdaptiveMaxPool2d(1)


    def forward(self, input, bb_box = None):
        x = self.features(input)
        x = self.pool_method(x)
        x = torch.flatten(x, 1)
        return F.normalize(x)


class VGG_Network(nn.Module):
    def __init__(self):
        super(VGG_Network, self).__init__()
        self.backbone = backbone_.vgg16(weights=backbone_.VGG16_Weights.DEFAULT).features
        self.pool_method =  nn.AdaptiveMaxPool2d(1)

    def forward(self, input, bb_box = None):
        x = self.backbone(input)
        x = self.pool_method(x).view(-1, 512)
        return F.normalize(x)

class InceptionV3_Network(nn.Module):
    def __init__(self, scale=128, use_att=False):
        super(InceptionV3_Network, self).__init__()
        self.use_att = use_att
        self.scale = scale
        backbone = backbone_.inception_v3(weights=backbone_.Inception_V3_Weights.DEFAULT)

        ## Extract Inception Layers ##
        self.Conv2d_1a_3x3 = backbone.Conv2d_1a_3x3
        self.Conv2d_2a_3x3 = backbone.Conv2d_2a_3x3
        self.Conv2d_2b_3x3 = backbone.Conv2d_2b_3x3
        self.Conv2d_3b_1x1 = backbone.Conv2d_3b_1x1
        self.Conv2d_4a_3x3 = backbone.Conv2d_4a_3x3
        self.Mixed_5b = backbone.Mixed_5b
        self.Mixed_5c = backbone.Mixed_5c
        self.Mixed_5d = backbone.Mixed_5d
        self.Mixed_6a = backbone.Mixed_6a
        self.Mixed_6b = backbone.Mixed_6b
        self.Mixed_6c = backbone.Mixed_6c
        self.Mixed_6d = backbone.Mixed_6d
        self.Mixed_6e = backbone.Mixed_6e

        self.Mixed_7a = backbone.Mixed_7a
        self.Mixed_7b = backbone.Mixed_7b
        self.Mixed_7c = backbone.Mixed_7c
        #self.pool_method =  nn.AdaptiveMaxPool2d(1) # as default
        self.pool_method =  nn.AdaptiveAvgPool2d(1) # as default

        if self.use_att:
            self.att_conv = nn.Conv2d(2048, 2048, kernel_size=1)

        if self.scale is not None:
            self.scale_fn = nn.Linear(2048, self.scale)

        self.finetuned_scale_fn = None

    def forward(self, x, finetune=False):
        # N x 3 x 299 x 299
        x = self.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 64 x 73 x 73
        x = self.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = F.max_pool2d(x, kernel_size=3, stride=2)
        # N x 192 x 35 x 35
        x = self.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.Mixed_7b(x)
        # N x 2048 x 8 x 8
        backbone_tensor = self.Mixed_7c(x)
        if self.use_att:
            att = self.att_conv(backbone_tensor)
            backbone_tensor = backbone_tensor + backbone_tensor * torch.softmax(att, dim=1)
        if finetune and self.finetuned_scale_fn is not None:
            feature = self.finetuned_scale_fn(backbone_tensor)
        else:
            feature = self.pool_method(backbone_tensor).view(-1, 2048)
            if self.scale is not None:
                feature = self.scale_fn(feature)

        return F.normalize(feature)
