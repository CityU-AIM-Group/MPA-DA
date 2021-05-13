import torch.nn as nn
import numpy as np
import torch
import torch.nn.functional as F
affine_par = True


def outS(i):
    i = int(i)
    i = (i + 1) / 2
    i = int(np.ceil((i + 1) / 2.0))
    i = (i + 1) / 2
    return i


def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False)  # change
        self.bn1 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False

        padding = dilation
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,  # change
                               padding=padding, bias=False, dilation=dilation)
        self.bn2 = nn.BatchNorm2d(planes, affine=affine_par)
        for i in self.bn2.parameters():
            i.requires_grad = False
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4, affine=affine_par)
        for i in self.bn3.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Classifier_Module(nn.Module):
    def __init__(self, inplanes, dilation_series, padding_series, num_classes):
        super(Classifier_Module, self).__init__()
        self.conv2d_list = nn.ModuleList()
        for dilation, padding in zip(dilation_series, padding_series):
            self.conv2d_list.append(
                nn.Conv2d(inplanes, num_classes, kernel_size=3, stride=1, padding=padding, dilation=dilation, bias=True))

        for m in self.conv2d_list:
            m.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv2d_list[0](x)
        for i in range(len(self.conv2d_list) - 1):
            out += self.conv2d_list[i + 1](x)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, affine=affine_par)
        for i in self.bn1.parameters():
            i.requires_grad = False
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1, ceil_mode=True)  # change
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)
        self.layer5 = self._make_pred_layer(Classifier_Module, 2048, [6, 12, 18, 24], [6, 12, 18, 24], num_classes)
        self.layer6 = nn.Sequential(
            nn.Conv2d(3, 1024, kernel_size=3, padding=1, dilation=1, bias=False),
            nn.BatchNorm2d(1024),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.layer7 = nn.Conv2d(1, 512, kernel_size=1, stride=1, bias=False)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, 0.01)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #        for i in m.parameters():
                #            i.requires_grad = False

    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion or dilation == 2 or dilation == 4:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion, affine=affine_par))
        for i in downsample._modules['1'].parameters():
            i.requires_grad = False
        layers = []
        layers.append(block(self.inplanes, planes, stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def _make_pred_layer(self, block, inplanes, dilation_series, padding_series, num_classes):
        return block(inplanes, dilation_series, padding_series, num_classes)

    def forward(self, x, y):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features_x = self.layer4(x)
        

        y = self.conv1(y)
        y = self.bn1(y)
        y = self.relu(y)
        y = self.maxpool(y)
        y = self.layer1(y)
        y = self.layer2(y)
        y = self.layer3(y)
        features_y = self.layer4(y)

        coarse_x = self.layer5(features_x)
        coarse_y = self.layer5(features_y)

        b, C, h, w = features_x.size()

        proj_key_x = features_x.view(b, C, -1).permute(0, 2, 1)
        proj_query_x = torch.sigmoid(coarse_x).view(b, 1, -1)
        prototype_x = torch.bmm(proj_query_x, proj_key_x)

        proj_key_y = features_y.view(b, C, -1).permute(0, 2, 1)
        proj_query_y = torch.sigmoid(coarse_y).view(b, 1, -1)
        prototype_y = torch.bmm(proj_query_y, proj_key_y)

        features_resize_x = features_x.view(b, C, -1)
        features_resize_y = features_y.view(b, C, -1)

        self_x = torch.bmm(prototype_x, features_resize_x).view(b, 1, h, w)
        self_x =  self_x / torch.max(self_x)
        #self_x = (self_x >= 0.5).float()
        cross_x = torch.bmm(prototype_y, features_resize_x).view(b, 1, h, w)
        cross_x = cross_x / torch.max(cross_x)
        #cross_x = (cross_x >= 0.5).float()

        self_y = torch.bmm(prototype_y, features_resize_y).view(b, 1, h, w)
        self_y = self_y / torch.max(self_y)
        #self_y = (self_y >= 0.5).float()
        cross_y = torch.bmm(prototype_x, features_resize_y).view(b, 1, h, w)
        cross_y = cross_y / torch.max(cross_y)
        #cross_y = (cross_y >= 0.5).float()
        #print(self_x.size(), self_y.size(), cross_x.size(), cross_y.size())

        refined_x = torch.cat([torch.sigmoid(coarse_x), self_x, cross_x], dim=1)
        refined_y = torch.cat([torch.sigmoid(coarse_y), self_y, cross_y], dim=1)
        #
        fine_x = self.layer6(refined_x)
        fine_y = self.layer6(refined_y)

        coarse_out_x = torch.sigmoid(F.interpolate(coarse_x, (256, 256), mode='bilinear', align_corners=True))
        coarse_out_y = torch.sigmoid(F.interpolate(coarse_y, (256, 256), mode='bilinear', align_corners=True))

        fine_out_x = torch.sigmoid(F.interpolate(fine_x, (256, 256), mode='bilinear', align_corners=True))
        fine_out_y = torch.sigmoid(F.interpolate(fine_y, (256, 256), mode='bilinear', align_corners=True))

        # prototype_x = prototype_x.permute(0, 2, 1)
        # proj_x = fine_x.view(b, 1, -1)
        # out_x = torch.bmm(prototype_x, proj_x).view(b, C, h, w)
        # reconstructed_x = torch.cat([out_x, fine_x], dim=1)
        #
        # prototype_y = prototype_y.permute(0, 2, 1)
        # proj_y = fine_y.view(b, 1, -1)
        # out_y = torch.bmm(prototype_y, proj_y).view(b, C, h, w)
        # reconstructed_y = torch.cat([out_y, fine_y], dim=1)

        # reconstructed_x = torch.sigmoid(F.interpolate(reconstructed_x, (256, 256), mode='bilinear', align_corners=True))
        # reconstructed_y = torch.sigmoid(F.interpolate(reconstructed_y, (256, 256), mode='bilinear', align_corners=True))

        #print(reconstructed_x.size(), reconstructed_y.size())

        return coarse_out_x, fine_out_x, coarse_out_y, fine_out_y
        #, reconstructed_x, reconstructed_y

    def get_1x_lr_params_NOscale(self):
        """
        This generator returns all the parameters of the net except for
        the last classification layer. Note that for each batchnorm layer,
        requires_grad is set to False in deeplab_resnet.py, therefore this function does not return
        any batchnorm parameter
        """
        b = []

        b.append(self.conv1)
        b.append(self.bn1)
        b.append(self.layer1)
        b.append(self.layer2)
        b.append(self.layer3)
        b.append(self.layer4)

        for i in range(len(b)):
            for j in b[i].modules():
                jj = 0
                for k in j.parameters():
                    jj += 1
                    if k.requires_grad:
                        yield k

    def get_10x_lr_params(self):
        """
        This generator returns all the parameters for the last layer of the net,
        which does the classification of pixel into classes
        """
        b = []
        b.append(self.layer5.parameters())
        b.append(self.layer6.parameters())
        #b.append(self.layer7.parameters())

        for j in range(len(b)):
            for i in b[j]:
                yield i

    def optim_parameters(self, args):
        return [{'params': self.get_1x_lr_params_NOscale(), 'lr': args.learning_rate},
                {'params': self.get_10x_lr_params(), 'lr': 10 * args.learning_rate}]


def MPA_model(num_classes=21):
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes)
    return model


if __name__ == '__main__':
    model = MPA_model(num_classes=1)
    s = torch.randn(1, 3, 256, 256)
    t = torch.randn(1, 3, 256, 256)
    coarse_s, fine_s, coarse_t, fine_t = model(s, t)
    print(coarse_s.size(), fine_s.size(), coarse_t.size(), fine_t.size())






