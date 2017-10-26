import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import os
import sys

class Pose_Estimation(nn.Module):

    def __init__(self, net_dict, layers, num_classes):

        super(Pose_Estimation, self).__init__()

        self.model0 = self._make_layer(net_dict[0])

        self.model1_1 = self._make_layer(net_dict[1][0])
        self.model1_2 = self._make_layer(net_dict[1][1])

        sefl.model2_1 = self._make_layer(net_dict[2][0])
        sefl.model2_2 = self._make_layer(net_dict[2][1])

        sefl.model3_1 = self._make_layer(net_dict[3][0])
        sefl.model3_2 = self._make_layer(net_dict[3][1])

        sefl.model4_1 = self._make_layer(net_dict[4][0])
        sefl.model4_2 = self._make_layer(net_dict[4][1])

        sefl.model5_1 = self._make_layer(net_dict[5][0])
        sefl.model5_2 = self._make_layer(net_dict[5][1])

        sefl.model6_1 = self._make_layer(net_dict[6][0])
        sefl.model6_2 = self._make_layer(net_dict[6][1])

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias in not None:
                    m.bias.data.zeros_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, net_dict, batch_norm=False, last_activity=True):

        layers = []
        length = len(net_dict) - 1
        for i in range(length):
            one_layer = net_dict[i]
            key = one_layer.keys()[0]
            v = one_layer[key]
            if 'pool' in key:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
            else:
                if batch_norm:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4], bias=False)
                    layers += [conv2d, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                    layers += [conv2d, nn.ReLU(inplace=True)]
        if last_activity:
            one_layer = net_dict[-1]
            key = one_layer.keys()[0]
            v = one_layer[key]

            if batch_norm:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4], bias=False)
                layers += [conv2d, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                layers += [conv2d, nn.ReLU(inplace=True)]
        else:
            one_layer = net_dict[-1]
            key = one_layer.keys()[0]
            v = one_layer[key]
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
            layers += [conv2d]
        return nn.Sequential(*layers)

    def forward(self, x):
        out0 = self.model0(x)

        out1_1 = self.model1_1(out0)
        out1_2 = self.model1_2(out0)
        out1 = torch.cat([out1_1, out1_2], 1)

        out2_1 = self.model2_1(out1)
        out2_2 = self.model2_2(out1)
        out2 = torch.cat([out2_1, out2_2], 1)

        out3_1 = self.model3_1(out2)
        out3_2 = self.model3_2(out2)
        out3 = torch.cat([out3_1, out3_2], 1)

        out4_1 = self.model4_1(out3)
        out4_2 = self.model4_2(out3)
        out4 = torch.cat([out4_1, out4_2], 1)

        out5_1 = self.model5_1(out4)
        out5_2 = self.model5_2(out4)
        out5 = torch.cat([out5_1, out5_2], 1)

        out6_1 = self.model6_1(out5)
        out6_2 = self.model6_2(out5)

        return out1_1, out1_2, out2_1, out2_2, out3_1, out3_2, out4_1, out4_2, out5_1, out5_2, out6_1, out6_2
