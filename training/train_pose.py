import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
sys.path.append('..')
import CocoFolder
import Mytransforms 
from BasicTool import adjust_learning_rate as adjust_learning_rate
from BasicTool import AverageMeter as AverageMeter
from BasicTool import save_checkpoint as save_checkpoint
from BasicTool import Config as Config
import pose_estimation

def parse():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str,
                        dest='config', help='to set the parameters')
    parser.add_argument('--gpu', default=[0], nargs='+', type=int,
                        dest='gpu', help='the gpu used')
    parser.add_argument('--pretrained', default=None,type=str,
                        dest='pretrained', help='the path of pretrained model')
    parser.add_argument('--root', default=None, type=str,
                        dest='root', help='the root of images')
    parser.add_argument('--train_dir', nargs='+', type=str,
                        dest='train_dir', help='the path of train file')
    parser.add_argument('--val_dir', default=None, nargs='+', type=str,
                        dest='val_dir', help='the path of val file')
    parser.add_argument('--num_classes', default=1000, type=int,
                        dest='num_classes', help='num_classes (default: 1000)')

    return parser.parse_args()

def construct_model(args):

    model = pose_estimation.PoseModel(num_point=19, num_vector=19, pretrained=True)
    # state_dict = torch.load(args.pretrained)['state_dict']
    # from collections import OrderedDict
    # new_state_dict = OrderedDict()
    # for k, v in state_dict.items():
        # name = k[7:]
        # new_state_dict[name] = v
    # model.load_state_dict(new_state_dict)
    # model.fc = nn.Linear(2048, 80)
    model = torch.nn.DataParallel(model, device_ids=args.gpu).cuda()

    return model

def train_val(model, args):

    traindir = args.train_dir
    valdir = args.val_dir
    print traindir, valdir

    config = Config(args.config)
    cudnn.benchmark = True
    
    train_loader = torch.utils.data.DataLoader(
            CocoFolder.CocoFolder(traindir, 8,
                Mytransforms.Compose([Mytransforms.RandomRotate(40),
                Mytransforms.RandomResizedCrop(368, 40),
                Mytransforms.RandomHorizontalFlip(),
            ])),
            batch_size=config.batch_size, shuffle=True,
            num_workers=config.workers, pin_memory=True)

    if config.test_interval != 0 and args.val_dir is not None:
        val_loader = torch.utils.data.DataLoader(
                CocoFolder.CocoFolder(valdir, 8,
                    Mytransforms.Compose([Mytransforms.Resize(400),
                    Mytransforms.CenterCrop(368),
                ])),
                batch_size=config.batch_size, shuffle=False,
                num_workers=config.workers, pin_memory=True)
    
    criterion = nn.MSELoss().cuda()
    
    optimizer = torch.optim.SGD(model.parameters(), config.base_lr,
                                momentum=config.momentum,
                                weight_decay=config.weight_decay)
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    topk = AverageMeter()
    
    end = time.time()
    iters = config.start_iters
    best_model = config.best_model
    learning_rate = config.base_lr

    model.train()
    while iters < config.max_iter:
    
        for i, (input, heatmap, vecmap, mask) in enumerate(train_loader):

            learning_rate = adjust_learning_rate(optimizer, iters, config.base_lr, policy=config.lr_policy, policy_parameter=config.policy_parameter)
            data_time.update(time.time() - end)

            heatmap = heatmap.cuda(async=True)
            vecmap = vecmap.cuda(async=True)
            input_var = torch.autograd.Variable(input)
            heatmap_var = torch.autograd.Variable(heatmap)
            vecmap_var = torch.autograd.Variable(vecmap)
            mask_var = torch.autograd.Variable(mask)

            vec1, heat1, vec2, heat2, vec3, heat3, vec4, heat4, vec5, heat5, vec6, heat6 = model(input_var, mask_var)
            loss1_1 = criterion(vec1, vecmap_var)
            loss1_2 = criterion(heat1, heatmap_var)
            loss2_1 = criterion(vec2, vecmap_var)
            loss2_2 = criterion(heat2, heatmap_var)
            loss3_1 = criterion(vec3, vecmap_var)
            loss3_2 = criterion(heat3, heatmap_var)
            loss4_1 = criterion(vec4, vecmap_var)
            loss4_2 = criterion(heat4, heatmap_var)
            loss5_1 = criterion(vec5, vecmap_var)
            loss5_2 = criterion(heat5, heatmap_var)
            loss6_1 = criterion(vec6, vecmap_var)
            loss6_2 = criterion(heat6, heatmap_var)
            
            loss = loss1_1 + loss1_2 + loss2_1 + loss2_2 + loss3_1 + loss3_2 + loss4_1 + loss4_2 + loss5_1 + loss5_2 + loss6_1 + loss6_2

            # prec1, preck = accuracy(output.data, label, topk=(1, config.topk))
            losses.update(loss.data[0], input.size(0))
            # top1.update(prec1[0], input.size(0))
            # topk.update(preck[0], input.size(0))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            batch_time.update(time.time() - end)
            end = time.time()
    
            iters += 1
            if iters % config.display == 0:
                print('Train Iteration: {0}\t'
                    'Time {batch_time.sum:.3f}s / 50iters, ({batch_time.avg:.3f})\t'
                    'Data load {data_time.sum:.3f}s / 50iters, ({data_time.avg:3f})\n'
                    'Learning rate = {1}\n'
                    'Loss = {loss.val:.4f} (ave = {loss.avg:.4f})\n'.format(
                    #'Prec@1 = {top1.val:.3f}% (ave = {top1.avg:.3f}%)\t'
                    #'Prec@{2} = {topk.val:.3f}% (ave = {topk.avg:.3f}%)\n'.format(
                    iters, learning_rate, batch_time=batch_time,
                    data_time=data_time, loss=losses))
                    #top1=top1, topk=topk))
                print time.strftime('%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n', time.localtime())
                batch_time.reset()
                data_time.reset()
                losses.reset()
                # top1.reset()
                # topk.reset()
    
            if config.test_interval != 0 and args.val_dir is not None and iters % config.test_interval == 0:

                model.eval()
                for i, (input, heatmap, vecmap, mask) in enumerate(val_loader):

                    heatmap = heatmap.cuda(async=True)
                    vecmap = vecmap.cuda(async=True)
                    input_var = torch.autograd.Variable(input, volatile=True)
                    heatmap_var = torch.autograd.Variable(heatmap, volatile=True)
                    vecmap_var = torch.autograd.Variable(vecmap, volatile=True)
                    mask_var = torch.autograd.Variable(mask, volatile=True)

                    vec1, heat1, vec2, heat2, vec3, heat3, vec4, heat4, vec5, heat5, vec6, heat6 = model(input_var, mask_var)
                    loss1_1 = criterion(vec1, vecmap_var)
                    loss1_2 = criterion(heat1, heatmap_var)
                    loss2_1 = criterion(vec2, vecmap_var)
                    loss2_2 = criterion(heat2, heatmap_var)
                    loss3_1 = criterion(vec3, vecmap_var)
                    loss3_2 = criterion(heat3, heatmap_var)
                    loss4_1 = criterion(vec4, vecmap_var)
                    loss4_2 = criterion(heat4, heatmap_var)
                    loss5_1 = criterion(vec5, vecmap_var)
                    loss5_2 = criterion(heat5, heatmap_var)
                    loss6_1 = criterion(vec6, vecmap_var)
                    loss6_2 = criterion(heat6, heatmap_var)
                    
                    loss = loss1_1 + loss1_2 + loss2_1 + loss2_2 + loss3_1 + loss3_2 + loss4_1 + loss4_2 + loss5_1 + loss5_2 + loss6_1 + loss6_2
                    # prec1, preck = accuracy(output.data, label, topk=(1, config.topk))
                    losses.update(loss.data[0], input.size(0))
                    # top1.update(prec1[0], input.size(0))
                    # topk.update(preck[0], input.size(0))
    
                batch_time.update(time.time() - end)
                end = time.time()
                is_best = losses.avg < best_model
                best_model = min(best_model, losses.avg)
                save_checkpoint({
                    'iter': iters,
                    'state_dict': model.state_dict(),
                    }, is_best, 'openpose_coco')
    
                print(
                    'Test Time {batch_time.sum:.3f}s, ({batch_time.avg:.3f})\t'
                    'Loss {loss.avg:.4f}\n'.format(
                    #'Prec@1 {top1.avg:.3f}%\t'
                    #'Prec@{0} {topk.avg:.3f}%\n'.format(
                    batch_time=batch_time, loss=losses))
                print time.strftime('%Y-%m-%d %H:%M:%S -----------------------------------------------------------------------------------------------------------------\n', time.localtime())
    
                batch_time.reset()
                losses.reset()
                # top1.reset()
                # topk.reset()
                
                model.train()
    
            if iters == config.max_iter:
                break


if __name__ == '__main__':

    args = parse()
    model = construct_model(args)
    train_val(model, args)
