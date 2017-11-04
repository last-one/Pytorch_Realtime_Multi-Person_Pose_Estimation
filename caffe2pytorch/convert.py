import caffe
from caffe.proto import caffe_pb2
import torch
import os
import sys
sys.path.append('..')
import pose_estimation
from utils import save_checkpoint as save_checkpoint

def load_caffe_model(deploy_path, model_path):

    caffe.set_mode_cpu()
    net = caffe.Net(deploy_path, model_path, caffe.TEST)

    return net

def load_pytorch_model():
    
    model = pose_estimation.PoseModel(num_point=19, num_vector=19, pretrained=True)

    return model

def convert(caffe_net, pytorch_net):

    caffe_keys = caffe_net.params.keys()
    pytorch_keys = pytorch_net.state_dict().keys()

    length_caffe = len(caffe_keys)
    length_pytorch = len(pytorch_keys)
    dic = {}
    L1 = []
    L2 = []
    _1 = []
    _2 = []
    for i in range(length_caffe):
        if 'L1' in caffe_keys[i]:
            L1.append(caffe_keys[i])
            if '_1' in pytorch_keys[2 * i]:
                _1.append(pytorch_keys[2 * i][:-7])
            else:
                _2.append(pytorch_keys[2 * i][:-7])
        elif 'L2' in caffe_keys[i]:
            L2.append(caffe_keys[i])
            if '_1' in pytorch_keys[2 * i]:
                _1.append(pytorch_keys[2 * i][:-7])
            else:
                _2.append(pytorch_keys[2 * i][:-7])
        else:
            dic[caffe_keys[i]] = pytorch_keys[2 * i][:-7]

    for info in zip(L1, _1):
        dic[info[0]] = info[1]
    for info in zip(L2, _2):
        dic[info[0]] = info[1]

    model_dict = pytorch_net.state_dict()
    from collections import OrderedDict
    weights_load = OrderedDict()
    for key in dic:
        caffe_key = key
        pytorch_key = dic[key]
        weights_load[pytorch_key + '.weight'] = torch.from_numpy(caffe_net.params[caffe_key][0].data)
        weights_load[pytorch_key + '.bias'] = torch.from_numpy(caffe_net.params[caffe_key][1].data)
    model_dict.update(weights_load)
    pytorch_net.load_state_dict(model_dict)
    save_checkpoint({
        'iter': 0,
        'state_dict': pytorch_net.state_dict(),
        }, True, 'caffe_model_coco')

if __name__ == '__main__':

    caffe_net = load_caffe_model('../caffe_model/coco/pose_deploy.prototxt', '../caffe_model/coco/pose_iter_440000.caffemodel')
    pytorch_net = load_pytorch_model()

    convert(caffe_net, pytorch_net)
