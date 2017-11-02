import torch
import torch.nn as nn
import os
import sys
import math
import torchvision.models as models

class Pose_Estimation(nn.Module):

    def __init__(self, net_dict, batch_norm=False):

        super(Pose_Estimation, self).__init__()

        self.model0 = self._make_layer(net_dict[0], batch_norm, True)

        self.model1_1 = self._make_layer(net_dict[1][0], batch_norm)
        self.model1_2 = self._make_layer(net_dict[1][1], batch_norm)

        self.model2_1 = self._make_layer(net_dict[2][0], batch_norm)
        self.model2_2 = self._make_layer(net_dict[2][1], batch_norm)

        self.model3_1 = self._make_layer(net_dict[3][0], batch_norm)
        self.model3_2 = self._make_layer(net_dict[3][1], batch_norm)

        self.model4_1 = self._make_layer(net_dict[4][0], batch_norm)
        self.model4_2 = self._make_layer(net_dict[4][1], batch_norm)

        self.model5_1 = self._make_layer(net_dict[5][0], batch_norm)
        self.model5_2 = self._make_layer(net_dict[5][1], batch_norm)

        self.model6_1 = self._make_layer(net_dict[6][0], batch_norm)
        self.model6_2 = self._make_layer(net_dict[6][1], batch_norm)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, net_dict, batch_norm=False, last_activity=False):

        layers = []
        length = len(net_dict) - 1
        for i in range(length):
            one_layer = net_dict[i]
            key = one_layer.keys()[0]
            v = one_layer[key]
            # print key, v
            if 'pool' in key:
                layers += [nn.MaxPool2d(kernel_size=v[0], stride=v[1], padding=v[2])]
            else:
                conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]

        if last_activity:
            one_layer = net_dict[-1]
            key = one_layer.keys()[0]
            v = one_layer[key]
            # print key, v
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v[1]), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
        else:
            one_layer = net_dict[-1]
            key = one_layer.keys()[0]
            v = one_layer[key]
            # print key, v
            conv2d = nn.Conv2d(in_channels=v[0], out_channels=v[1], kernel_size=v[2], stride=v[3], padding=v[4])
            layers += [conv2d]
        return nn.Sequential(*layers)

    def forward(self, x, mask):
        out0 = self.model0(x)

        out1_1 = self.model1_1(out0)
        out1_2 = self.model1_2(out0)
        out1 = torch.cat([out1_1, out1_2, out0], 1)
        out1_vec_mask = out1_1 * mask.expand_as(out1_1)
        out1_heat_mask = out1_2 * mask.expand_as(out1_2)

        out2_1 = self.model2_1(out1)
        out2_2 = self.model2_2(out1)
        out2 = torch.cat([out2_1, out2_2, out0], 1)
        out2_vec_mask = out2_1 * mask.expand_as(out2_1)
        out2_heat_mask = out2_2 * mask.expand_as(out2_2)

        out3_1 = self.model3_1(out2)
        out3_2 = self.model3_2(out2)
        out3 = torch.cat([out3_1, out3_2, out0], 1)
        out3_vec_mask = out3_1 * mask.expand_as(out3_1)
        out3_heat_mask = out3_2 * mask.expand_as(out3_2)

        out4_1 = self.model4_1(out3)
        out4_2 = self.model4_2(out3)
        out4 = torch.cat([out4_1, out4_2, out0], 1)
        out4_vec_mask = out4_1 * mask.expand_as(out4_1)
        out4_heat_mask = out4_2 * mask.expand_as(out4_2)

        out5_1 = self.model5_1(out4)
        out5_2 = self.model5_2(out4)
        out5 = torch.cat([out5_1, out5_2, out0], 1)
        out5_vec_mask = out5_1 * mask.expand_as(out5_1)
        out5_heat_mask = out5_2 * mask.expand_as(out5_2)

        out6_1 = self.model6_1(out5)
        out6_2 = self.model6_2(out5)
        out6_vec_mask = out6_1 * mask.expand_as(out6_1)
        out6_heat_mask = out6_2 * mask.expand_as(out6_2)

        return out1_vec_mask, out1_heat_mask, out2_vec_mask, out2_heat_mask, out3_vec_mask, out3_heat_mask, out4_vec_mask, out4_heat_mask, out5_vec_mask, out5_heat_mask, out6_vec_mask, out6_heat_mask

def PoseModel(num_point, num_vector, num_stages=6, batch_norm=False, pretrained=False):

    net_dict = []
    block0 = [{'conv1_1': [3, 64, 3, 1, 1]}, {'conv1_2': [64, 64, 3, 1, 1]}, {'pool1': [2, 2, 0]},
            {'conv2_1': [64, 128, 3, 1, 1]}, {'conv2_2': [128, 128, 3, 1, 1]}, {'pool2': [2, 2, 0]},
            {'conv3_1': [128, 256, 3, 1, 1]}, {'conv3_2': [256, 256, 3, 1, 1]}, {'conv3_3': [256, 256, 3, 1, 1]}, {'conv3_4': [256, 256, 3, 1, 1]}, {'pool3': [2, 2, 0]},
            {'conv4_1': [256, 512, 3, 1, 1]}, {'conv4_2': [512, 512, 3, 1, 1]}, {'conv4_3_cpm': [512, 256, 3, 1, 1]}, {'conv4_4_cpm': [256, 128, 3, 1, 1]}]
    net_dict.append(block0)

    block1 = [[], []]
    in_vec = [0, 128, 128, 128, 128, 512, num_vector * 2]
    in_heat = [0, 128, 128, 128, 128, 512, num_point]
    for i in range(1, 6):
        if i < 4:
            block1[0].append({'conv{}_stage1_vec'.format(1) :[in_vec[i], in_vec[i + 1], 3, 1, 1]})
            block1[1].append({'conv{}_stage1_heat'.format(1):[in_heat[i], in_heat[i + 1], 3, 1, 1]})
        else:
            block1[0].append({'conv{}_stage1_vec'.format(1):[in_vec[i], in_vec[i + 1], 1, 1, 0]})
            block1[1].append({'conv{}_stage1_heat'.format(1):[in_heat[i], in_heat[i + 1], 1, 1, 0]})
    net_dict.append(block1)

    in_vec_1 = [0, 128 + num_point + num_vector * 2, 128, 128, 128, 128, 128, 128, num_vector * 2]
    in_heat_1 = [0, 128 + num_point + num_vector * 2, 128, 128, 128, 128, 128, 128, num_point]
    for j in range(2, num_stages + 1):
        blocks = [[], []]
        for i in range(1, 8):
            if i < 6:
                blocks[0].append({'conv{}_stage{}_vec'.format(i, j):[in_vec_1[i], in_vec_1[i + 1], 7, 1, 3]})
                blocks[1].append({'conv{}_stage{}_heat'.format(i, j):[in_heat_1[i], in_heat_1[i + 1], 7, 1, 3]})
            else:
                blocks[0].append({'conv{}_stage{}_vec'.format(i, j):[in_vec_1[i], in_vec_1[i + 1], 1, 1, 0]})
                blocks[1].append({'conv{}_stage{}_heat'.format(i, j):[in_heat_1[i], in_heat_1[i + 1], 1, 1, 0]})
        net_dict.append(blocks)

    model = Pose_Estimation(net_dict, batch_norm)

    if pretrained:
        parameter_num = 10
        if batch_norm:
            vgg19 = models.vgg19_bn(pretrained=True)
            parameter_num *= 6
        else:
            vgg19 = models.vgg19(pretrained=True)
            parameter_num *= 2
        vgg19_state_dict = vgg19.state_dict()
        vgg19_keys = vgg19_state_dict.keys()

        model_dict = model.state_dict()
        from collections import OrderedDict
        weights_load = OrderedDict()
        for i in range(parameter_num):
            weights_load[model.state_dict().keys()[i]] = vgg19_state_dict[vgg19_keys[i]]
        model_dict.update(weights_load)
        model.load_state_dict(model_dict)

    return model

if __name__ == '__main__':

    print PoseModel(19, 6, True, True)
